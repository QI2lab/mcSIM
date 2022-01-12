/**
 * This example plugin pops up a dialog box that says "Hello, world!".
 *
 * See https://micro-manager.org/wiki/Version_2.0_Plugins#Defining_a_Plugin for more help
 * 
 * Copyright University of California
 * 
 * LICENSE:      This file is distributed under the BSD license.
 *               License text is included with the source distribution.
 *
 *               This file is distributed in the hope that it will be useful,
 *               but WITHOUT ANY WARRANTY; without even the implied warranty
 *               of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 *               IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 *               CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *               INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
 *
 *
 *
 */

package org.micromanager.plugins.SIM;

import javax.swing.JOptionPane;

import mmcorej.CMMCore;

import org.micromanager.MenuPlugin;
import org.micromanager.Studio;

import org.scijava.plugin.Plugin;
import org.scijava.plugin.SciJavaPlugin;

@Plugin(type = MenuPlugin.class)
public class SIM implements SciJavaPlugin, MenuPlugin {
   // Provides access to the MicroManager API.
   private Studio studio_;
   private org.micromanager.plugins.SIM.SIMFrame frame_;
   private Runtime runtime_;
   private Process process_;

   @Override
   public void setContext(Studio studio) {
      studio_ = studio;
   }

   /**
    * This method is called when the plugin's menu option is selected.
    */
   @Override
   public void onPluginSelected() {
      if (frame_ == null) {
         frame_ = new org.micromanager.plugins.SIM.SIMFrame(studio_);
         // JOptionPane.showMessageDialog(null, "Hello, world!", "Hello world!", JOptionPane.PLAIN_MESSAGE);
      }

      frame_.setVisible(true);
   }

   /**
    * This method determines which sub-menu of the Plugins menu we are placed
    * into.
    */
   @Override
   public String getSubMenu() {
      return "";
   }

   @Override
   public String getName() {
      return "DMD SIM control";
   }

   @Override
   public String getHelpText() {
      return "Automate DMD control for displaying SIM patterns while exploring sample.";
   }

   @Override
   public String getVersion() {
      return "0.0";
   }

   @Override
   public String getCopyright() {
      return "Peter T. Brown, Arizona State University, 2021";
   }
}
