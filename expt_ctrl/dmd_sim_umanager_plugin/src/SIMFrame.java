package org.micromanager.plugins.SIM;

import org.micromanager.Studio;
import org.micromanager.UserProfile;
import org.micromanager.propertymap.MutablePropertyMapView;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.IOException;

import java.io.InputStreamReader;


public class SIMFrame extends JFrame {

    private Studio studio_;
    private UserProfile profile_;
    private MutablePropertyMapView settings_;

    private String condaEnvName_;
    private String condaEnvProfileKey_ = "condaEnvName";
    private String scriptPath_;
    private String scriptProfileKey_ = "pythonScriptPath";
    private String color_;
    private int phase_;
    private int angle_;
    private String mode_;
    private boolean single_pattern_mode_;

    private JPanel panelMain;
    private JTextArea errorLabel;
    private JTextField textFieldScript;
    private JTextField textFieldEnv;
    private JComboBox comboBoxAngle;
    private JComboBox comboBoxPhase;
    private JComboBox comboBoxMode;
    private JComboBox comboBoxColor;
    private JComboBox comboBoxSinglePattern;

    public SIMFrame(Studio app){
        super("DMD-SIM explorer");
        studio_ = app;
        profile_ = app.getUserProfile();
        settings_ = profile_.getSettings(this.getClass());

        // main frame
        setBounds(100, 100, 900, 500);
//        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // main panel
        panelMain = new JPanel();
        panelMain.setLayout(new GridLayout(10, 2, 0, 10));

        // python script path text pane
        textFieldScript = new JTextField(100);
        textFieldScript.setToolTipText("Enter path to python script");
        textFieldScript.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                warn();
            }
            public void removeUpdate(DocumentEvent e) {
                warn();
            }
            public void insertUpdate(DocumentEvent e) {
                warn();
            }

            public void warn() {
                setScriptPath_(textFieldScript.getText());
            }
        });

        // Conda environment path text pane
        textFieldEnv = new JTextField(100);
        textFieldEnv.setToolTipText("Enter name of conda environment");
        textFieldEnv.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                warn();
            }
            public void removeUpdate(DocumentEvent e) {
                warn();
            }
            public void insertUpdate(DocumentEvent e) {
                warn();
            }

            public void warn() {
                setCondaEnvName_(textFieldEnv.getText());
            }
        });

        // add color selector
        String [] colorStrings = {"blue", "green", "red", "nir"};
        comboBoxColor = new JComboBox(colorStrings);
        comboBoxColor.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                setColor_((String) comboBoxColor.getSelectedItem());

            }
        });
        comboBoxColor.setSelectedIndex(0);

        // add angle selector
        String [] angles = {"0", "1", "2"};
        comboBoxAngle = new JComboBox(angles);
        comboBoxAngle.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                setAngle_(Integer.parseInt((String) comboBoxAngle.getSelectedItem()));
            }
        });
        comboBoxAngle.setSelectedIndex(0);

        // add phase selector
        String [] phases = {"0", "1", "2"};
        comboBoxPhase = new JComboBox(phases);
        comboBoxPhase.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                setPhase_(Integer.parseInt((String) comboBoxPhase.getSelectedItem()));
            }
        });
        comboBoxPhase.setSelectedIndex(0);

        // mode selector
        String [] modes = {"sim", "widefield", "affine"};
        comboBoxMode = new JComboBox(modes);
        comboBoxMode.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                setMode_((String) comboBoxMode.getSelectedItem());
            }
        });
        comboBoxMode.setSelectedIndex(0);

        // single pattern mode selector
        String [] choices = {"false", "true"};
        comboBoxSinglePattern = new JComboBox(choices);
        comboBoxSinglePattern.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                setSingle_pattern_mode_( ((int) comboBoxSinglePattern.getSelectedIndex()) == 1);
            }
        });
        comboBoxSinglePattern.setSelectedIndex(0);

        // add display button
        JButton displayButton = new JButton("Display");
        displayButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {

                Runtime r = Runtime.getRuntime();
                Process p;
                try {
                    StringBuilder commandBuilder = new StringBuilder("cmd /c conda activate ");
                    commandBuilder.append(condaEnvName_);
                    commandBuilder.append(" & python ");
                    commandBuilder.append(scriptPath_);
                    commandBuilder.append(" ");
                    commandBuilder.append(color_);
                    commandBuilder.append(" -m ");
                    commandBuilder.append(mode_);
                    if (single_pattern_mode_) {
                        commandBuilder.append(" -s");
                        commandBuilder.append(" -a");
                        commandBuilder.append(angle_);
                        commandBuilder.append(" -p");
                        commandBuilder.append(phase_);
                    }

                    p = r.exec(commandBuilder.toString());
                    BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));
                    BufferedReader stdError = new BufferedReader(new InputStreamReader(p.getErrorStream()));

                    StringBuilder resultBuilder = new StringBuilder("");
                    StringBuilder errBuilder = new StringBuilder("");

                    String s;
                    while ((s = stdInput.readLine()) != null){
                        resultBuilder.append(s);
                    }

                    while ((s = stdError.readLine()) != null){
                        errBuilder.append(s);
                    }

                    errorLabel.setText(errBuilder.toString());

                    //JOptionPane.showMessageDialog(null, errBuilder.toString(), "Script results", JOptionPane.PLAIN_MESSAGE);

                } catch (IOException ioException) {
                    ioException.printStackTrace();
                }

            }
        });

        // add button to save strings to user profile
        JButton buttonSaveString = new JButton("Save paths in user profile");
        buttonSaveString.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                settings_.putString(condaEnvProfileKey_, condaEnvName_);
                settings_.putString(scriptProfileKey_, scriptPath_);
            }
        });

        // show error message from script
        errorLabel = new JTextArea();
        errorLabel.setText("");
        errorLabel.setLineWrap(true);

        // add controls to JPanel
        // getContentPane().add(textFieldScript);
        panelMain.add(new Label("Enter path to conda environment"));
        panelMain.add(textFieldEnv);

        panelMain.add(new Label("Enter path to python script"));
        panelMain.add(textFieldScript);

        panelMain.add(new Label(""));
        panelMain.add(buttonSaveString);

        panelMain.add(new Label("laser"));
        panelMain.add(comboBoxColor);

        panelMain.add(new Label("mode"));
        panelMain.add(comboBoxMode);

        panelMain.add(new Label("angle index"));
        panelMain.add(comboBoxAngle);

        panelMain.add(new Label("phase index"));
        panelMain.add(comboBoxPhase);

        panelMain.add(new Label("Single pattern mode"));
        panelMain.add(comboBoxSinglePattern);

        panelMain.add(new Label(""));
        panelMain.add(displayButton);

        // error results spans two columns
        GridBagConstraints c = new GridBagConstraints();
        c.fill = GridBagConstraints.BOTH;
        c.gridwidth = 3;
        panelMain.add(errorLabel, c);

        // add JPanel to JFrame
        add(panelMain);
//        pack();

        // grab string values from settings
        String envName = settings_.getString(condaEnvProfileKey_, "");
        textFieldEnv.setText(envName);
        String scriptPath = settings_.getString(scriptProfileKey_, "");
        textFieldScript.setText(scriptPath);

    }

    public void setScriptPath_(String scriptPath_) {
        this.scriptPath_ = scriptPath_;
    }

    public void setCondaEnvName_(String condaEnvName_) {
        this.condaEnvName_ = condaEnvName_;
    }

    public void setColor_(String color_) {
        this.color_ = color_;
    }

    public void setPhase_(int phase_) {
        this.phase_ = phase_;
    }

    public void setAngle_(int angle_) {
        this.angle_ = angle_;
    }

    public void setMode_(String mode_) {
        this.mode_ = mode_;
    }

    public void setSingle_pattern_mode_(boolean single_pattern_mode_) {
        this.single_pattern_mode_ = single_pattern_mode_;
    }
}
