"""
Control Phantom camera using their SDK and ctypes
"""

import numpy as np
import ctypes as ct
from pathlib import Path

# Codes defined in PhFile.h
# generated these programmatically
MIFILE_RAWCINE = 0
MIFILE_CINE = 1
MIFILE_JPEGCINE = 2
MIFILE_AVI = 3
MIFILE_TIFCINE = 4
MIFILE_MPEG = 5
MIFILE_MXFPAL = 6
MIFILE_MXFNTSC = 7
MIFILE_QTIME = 8
MIFILE_MP4 = 9
SIFILE_WBMP24 = -1
SIFILE_WBMP4 = -3
SIFILE_OBMP4 = -4
SIFILE_OBMP8 = -5
SIFILE_OBMP24 = -6
SIFILE_TIF1 = -7
SIFILE_TIF8 = -8
SIFILE_TIF12 = -9
SIFILE_TIF16 = -10
SIFILE_PCX24 = -13
SIFILE_TGA8 = -14
SIFILE_TGA16 = -15
SIFILE_TGA32 = -17
SIFILE_LEAD1JTIF = -21
SIFILE_LEAD2JTIF = -22
SIFILE_JPEG = -23
SIFILE_JTIF = -24
SIFILE_DNG = -26
SIFILE_DPX = -27
SIFILE_EXR = -28
TF_LT = 0
TF_UT = 1
TF_SMPTE = 2
PPT_FULL = 0
PPT_DATE_ONLY = 0
PPT_TIME_ONLY = 0
PPT_FRAC_ONLY = 0
PPT_ATTRIB_ONLY = 0
UC_VIEW = 1
UC_SAVE = 2
GCI_CFA = 0
GCI_FRAMERATE = 1
GCI_EXPOSURE = 2
GCI_AUTOEXPOSURE = 3
GCI_REALBPP = 4
GCI_CAMERASERIAL = 5
GCI_HEADSERIAL0 = 6
GCI_HEADSERIAL1 = 7
GCI_HEADSERIAL2 = 8
GCI_HEADSERIAL3 = 9
GCI_TRIGTIMESEC = 10
GCI_TRIGTIMEFR = 11
GCI_ISFILECINE = 12
GCI_IS16BPPCINE = 13
GCI_ISCOLORCINE = 14
GCI_ISMULTIHEADCINE = 15
GCI_EXPOSURENS = 16
GCI_EDREXPOSURENS = 17
GCI_FRAMEDELAYNS = 19
GCI_FROMFILETYPE = 20
GCI_COMPRESSION = 21
GCI_CAMERAVERSION = 22
GCI_ROTATE = 23
GCI_IMWIDTH = 24
GCI_IMHEIGHT = 25
GCI_IMWIDTHACQ = 26
GCI_IMHEIGHTACQ = 27
GCI_POSTTRIGGER = 28
GCI_IMAGECOUNT = 29
GCI_TOTALIMAGECOUNT = 30
GCI_TRIGFRAME = 31
GCI_AUTOEXPLEVEL = 32
GCI_AUTOEXPTOP = 33
GCI_AUTOEXPLEFT = 34
GCI_AUTOEXPBOTTOM = 35
GCI_AUTOEXPRIGHT = 36
GCI_RECORDINGTIMEZONE = 37
GCI_FIRSTIMAGENO = 38
GCI_FIRSTMOVIEIMAGE = 39
GCI_CINENAME = 40
GCI_TIMEFORMAT = 41
GCI_MARKIN = 42
GCI_MARKOUT = 43
GCI_FROMFILENAME = 44
GCI_PARTITIONNO = 45
GCI_GPS = 46
GCI_UUID = 47
GCI_RECBPP = 48
GCI_MAGSERIAL = 49
GCI_CSSERIAL = 50
GCI_SENSOR = 51
GCI_SENSORMODE = 52
GCI_FRPSTEPS = 100
GCI_FRP1X = 101
GCI_FRP1Y = 102
GCI_FRP2X = 103
GCI_FRP2Y = 104
GCI_FRP3X = 105
GCI_FRP3Y = 106
GCI_FRP4X = 107
GCI_FRP4Y = 108
GCI_WRITEERR = 109
GCI_LENSDESCRIPTION = 110
GCI_LENSAPERTURE = 111
GCI_LENSFOCALLENGTH = 112
GCI_WB = 200
GCI_WBVIEW = 201
GCI_WBISMETA = 230
GCI_BRIGHT = 202
GCI_CONTRAST = 203
GCI_GAINR = 232
GCI_GAING = 233
GCI_GAINB = 234
GCI_GAMMA = 204
GCI_GAMMAR = 223
GCI_GAMMAB = 224
GCI_SATURATION = 205
GCI_HUE = 206
GCI_FLIPH = 207
GCI_FLIPV = 208
GCI_FILTERCODE = 209
GCI_IMFILTER = 210
GCI_INTALGO = 211
GCI_PEDESTALR = 212
GCI_PEDESTALG = 213
GCI_PEDESTALB = 214
GCI_FLARE = 225
GCI_CHROMA = 226
GCI_TONE = 227
GCI_ENABLEMATRICES = 228
GCI_USERMATRIX = 229
GCI_CALIBMATRIX = 231
GCI_RESAMPLEACTIVE = 215
GCI_RESAMPLEWIDTH = 216
GCI_RESAMPLEHEIGHT = 217
GCI_CROPACTIVE = 218
GCI_CROPRECTANGLE = 219
GCI_OFFSET16_8 = 220
GCI_GAIN16_8 = 221
GCI_MAXGAIN16_8 = 242
GCI_DEMOSAICINGFUNCPTR = 222
GCI_WBTEMPERATURE = 235
GCI_WBCOLCOMP = 236
GCI_OPTICALFILTER = 237
GCI_CALIBINFO = 238
GCI_GAMMARCH = 239
GCI_GAMMAGCH = 240
GCI_GAMMABCH = 241
GCI_SUPPORTSTOE = 243
GCI_TOE = 244
GCI_SUPPORTSLOGMODE = 245
GCI_LOGMODE = 246
GCI_EI = 250
GCI_CAMERAMODEL = 251
GCI_MARKSAT = 252
GCI_VFLIPVIEWACTIVE = 300
GCI_MAXIMGSIZE = 400
GCI_FRPIMGNOARRAY = 450
GCI_FRPRATEARRAY = 451
GCI_FRPSHAPEARRAY = 452
GCI_TRIGTC = 470
GCI_TRIGTCU = 471
GCI_PBRATE = 472
GCI_TCRATE = 473
GCI_DFRAMERATE = 480
GCI_SAVERANGE = 500
GCI_SAVEFILENAME = 501
GCI_SAVEFILETYPE = 502
GCI_SAVE16BIT = 503
GCI_SAVEPACKED = 504
GCI_SAVEXML = 505
GCI_SAVESTAMPTIME = 506
GCI_SAVEDEFFOLDER = 507
GCI_SAVEDEFSNAPSHOTFOLDER = 508
GCI_SAVEDECIMATION = 1035
GCI_SAVEAVIFRAMERATE = 520
GCI_SAVEAVICOMPRESSORLIST = 521
GCI_SAVEAVICOMPRESSORINDEX = 522
GCI_SAVEDPXLSB = 530
GCI_SAVEDPXTO10BPS = 531
GCI_SAVEDPXDATAPACKING = 532
GCI_SAVEDPX10BITLOG = 533
GCI_SAVEDPXEXPORTLOGLUT = 534
GCI_SAVEDPX10BITLOGREFWHITE = 535
GCI_SAVEDPX10BITLOGREFBLACK = 536
GCI_SAVEDPX10BITLOGGAMMA = 537
GCI_SAVEDPX10BITLOGFILMGAMMA = 538
GCI_SAVEQTPLAYBACKRATE = 550
GCI_SAVECCIQUALITY = 560
GCI_UNCALIBRATEDIMAGE = 600
GCI_NOPROCESSING = 601
GCI_BADPIXELREPAIR = 602
GCI_CINEDESCRIPTION = 603
GCI_RDTYPESCNT = 700
GCI_RDNAME = 701
MAXCOMPRCNT = 64
SSC_NAME = 1
SSC_EXPLORE = 2
SSC_CLEAN = 3
SSC_SETTINGS = 4
ERR_BadCine = -2000
ERR_UnsupportedFormat = -2002
ERR_InsufficientAlloc = -2003
ERR_NotInRange = -2004
ERR_SaveAvi = -2005
ERR_Encoder = -2006
ERR_UnsupportedTiffFormat = -2007
ERR_SplitQuartersUnsupported = -2009
ERR_NotACineFile = -2010
ERR_FileOpen = -2012
ERR_FileRead = -2013
ERR_FileWrite = -2014
ERR_FileSeek = -2015
ERR_DecompressImage = -2016
ERR_CineVerNewer = -2017
ERR_NotSupported = -2018
ERR_FileInUse = -2019
ERR_CannotBuildGraph = -2021
ERR_AuxDataNotFound = -2023
ERR_NotEnoughMemory = -2024
ERR_MpegReaderNotFound = -2025
ERR_FunctionNotFound = -2028
ERR_CineInUse = -2029
ERR_SaveCineBufferFull = -2030
ERR_NoCineSaveInProgress = -2031
ERR_cfls = -2032
ERR_cfread = -2033
ERR_cfrm = -2034
ERR_ANNOTATION_TOO_BIG = -2040
OFN_MULTISELECT = 1
SFH_HEAD0 = 1
SFH_HEAD1 = 2
SFH_HEAD2 = 4
SFH_HEAD3 = 8
SFH_ALLHEADS = 0

# codes defined in PhCon.h
PHCONHEADERVERSION = 800
MAXCAMERACNT = 63
MAXERRMESS = 256
MAXIPSTRSZ = 16
DFOURGIG = 4294967296
PI = 3
CINE_DEFAULT = -2
CINE_CURRENT = -1
CINE_PREVIEW = 0
CINE_FIRST = 1
gsHasMechanicalShutter = 1025
gsHasBlackLevel4 = 1027
gsHasCardFlash = 1051
gsHas10G = 2000
gsHasV2AutoExposure = 2001
gsHasV2LockAtTrigger = 2002
gsSupportsInternalBlackRef = 1026
gsSupportsImageTrig = 1040
gsSupportsCardFlash = 1050
gsSupportsMagazine = 8193
gsSupportsHQMode = 8194
gsSupportsGenlock = 8195
gsSupportsEDR = 8196
gsSupportsAutoExposure = 8197
gsSupportsTurbo = 8198
gsSupportsBurstMode = 8199
gsSupportsShutterOff = 8200
gsSupportsDualSDIOutput = 8201
gsSupportsRecordingCines = 8202
gsSupportsV444 = 8203
gsSupportsInterlacedSensor = 8204
gsSupportsRampFRP = 8205
gsSupportsOffGainCorrections = 8206
gsSupportsFRP = 8207
gsSupportedVideoSystems = 8208
gsSupportsRemovableSSD = 8209
gsSupportedAuxSignalFunctions = 8210
gsSupportsGps = 8211
gsSupportsVideoSync = 8212
gsSupportsTimeCodeOutSignal = 8213
gsSupportsQuietMode = 8214
gsSupportsPreTriggerMemGate = 8215
gsSupportsV4K = 8216
gsSupportsAnamorphicDesqRatio = 8217
gsSupportsAudio = 8218
gsSupportsProRes = 8219
gsSupportsV3G = 8220
gsSupportsProgIO = 8221
gsSupportsSyncToTrigger = 8222
gsSupportsSensorModes = 8223
gsSupportsBattery = 8224
gsSupportsExpIndex = 8225
gsSupportsHvTrigger = 8226
gsSupports10G = 8227
gsSupportsOnCamCtrl = 8228
gsSupportsPaMode = 8229
gsSensorTemperature = 1028
gsCameraTemperature = 1029
gsThermoElectricPower = 1030
gsSensorTemperatureThreshold = 1031
gsCameraTemperatureThreshold = 1032
gsVideoPlayCine = 1033
gsVideoPlaySpeed = 1034
gsVideoOutputConfig = 1035
gsMechanicalShutter = 1036
gsImageTrigThreshold = 1041
gsImageTrigAreaPercentage = 1042
gsImageTrigSpeed = 1043
gsImageTrigMode = 1044
gsImageTrigRect = 1045
gsAutoProgress = 1046
gsAutoBlackRef = 1047
gsCardFlashError = 1054
gsIPAddress = 1070
gsEthernetAddress = 1055
gsEthernetMask = 1056
gsEthernetBroadcast = 1057
gsEthernetGateway = 1058
gsEthernet10GAddress = 1093
gsEthernet10GMask = 1094
gsEthernet10GBroadcast = 1095
gsEthernetDefaultAddress = 1096
gsLensFocus = 1059
gsLensAperture = 1060
gsLensApertureRange = 1061
gsLensDescription = 1062
gsLensFocusInProgress = 1063
gsLensFocusAtLimit = 1064
gsGenlock = 1065
gsGenlockStatus = 1066
gsTurboMode = 1068
gsModel = 1069
gsMaxPartitionCount = 1071
gsAuxSignal = 1072
gsTimeCodeOutSignal = 1073
gsQuiet = 1074
gsBaseEI = 1075
gsVFMode = 1076
gsAnamorphicDesqRatio = 1077
gsAudioEnable = 1078
gsClockPeriod = 1079
gsPortCount = 1080
gsTriggerEdgeAndVoltage = 1081
gsTriggerFilter = 1082
gsTriggerDelay = 1083
gsHeadSerial = 1085
gsHeadTemperature = 1086
gsSensorModesList = 1087
gsSensorMode = 1088
gsExpIndex = 1090
gsExpIndexPresets = 1091
gsBatteryEnable = 1100
gsBatteryCaptureEnable = 1101
gsBatteryPreviewEnable = 1102
gsBatteryWtrRuntime = 1103
gsBatteryVoltage = 1104
gsBatteryState = 1105
gsBatteryMode = 1106
gsBatteryArmDelay = 1107
gsBatteryPrevRuntime = 1108
gsBatteryPwrOffDelay = 1109
gsBatteryReadyGate = 1110
gsUpTime = 1120
gsStartUps = 1121
gsOnCamCtrlEnable = 1130
gsOverlayTypeStyle = 1136
gsAux1Signal = 3008
gsAux2Signal = 3009
gsAux3Signal = 3010
gsSupportedAux1SignalFunctions = 9020
gsSupportedAux2SignalFunctions = 9021
gsSupportedAux3SignalFunctions = 9022
gsSigCount = 10001
gsSigSelect = 10002
gsPulseProc = 10003
gsHasPulseProc = 10004
gsSigNameFmw = 20001
gsSigName = 20002
cgsVideoTone = 4097
cgsName = 4098
cgsVideoMarkIn = 4099
cgsVideoMarkOut = 4100
cgsIsRecorded = 4101
cgsHqMode = 4102
cgsBurstCount = 4103
cgsBurstPeriod = 4104
cgsLensDescription = 4105
cgsLensAperture = 4106
cgsLensFocalLength = 4107
cgsShutterOff = 4108
cgsTriggerTime = 4109
cgsTrigTC = 4110
cgsPbRate = 4111
cgsTcRate = 4112
cgsGps = 4113
cgsUuid = 4114
cgsModel = 4120
cgsAutoExpComp = 4200
cgsDescription = 4210
GV_CAMERA = 1
GV_FIRMWARE = 2
GV_FPGA = 3
GV_PHCON = 4
GV_CFA = 5
GV_KERNEL = 6
GV_MAGAZINE = 7
GV_FIRMWAREPACK = 8
GV_HEAD_FIRMWARE = 9
GV_HEAD_FPGA = 10
GV_HEAD_FIRMWAREPACK = 11
GV_VSHUTTER_FIRMWARE = 12
GV_STREAMER_FIRMWARE = 13
AUXSIG_STROBE = 0
AUXSIG_IRIGOUT = 1
AUXSIG_EVENT = 2
AUXSIG_MEMGATE = 3
AUXSIG_FSYNC = 4
AUX1SIG_STROBE = 0
AUX1SIG_EVENT = 1
AUX1SIG_MEMGATE = 2
AUX1SIG_FSYNC = 3
AUX2SIG_READY = 0
AUX2SIG_STROBE = 1
AUX2SIG_AES_EBU_OUT = 2
AUX3SIG_IRIGOUT = 0
AUX3SIG_STROBE = 1
TIMECODEOUTSIG_IRIG = 0
TIMECODEOUTSIG_SMPTE = 1
DO_IGNORECAMERAS = 1
DO_PERCAMERAACQPARAMS = 2
DO_SUPPRESSOFFLINESTAMP = 7
DO_ANGLEUNITS = 9
NVCR_CONT_REC = 0
NVCR_APV = 0
NVCR_REC_ONCE = 0
SYNC_INTERNAL = 0
SYNC_EXTERNAL = 1
SYNC_LOCKTOIRIG = 2
SYNC_LOCKTOVIDEO = 3
SYNC_SYNCTOTRIGGER = 5
CFA_NONE = 0
CFA_VRI = 1
CFA_VRIV6 = 2
CFA_BAYER = 3
CFA_BAYERFLIP = 4
CFA_BAYERFLIPB = 5
CFA_BAYERFLIPH = 6
CFA_MASK = 0
TL_GRAY = 0
TR_GRAY = 0
BL_GRAY = 0
BR_GRAY = 0
ALLHEADS_GRAY = 0
VIDEOSYS_NTSC = 0
VIDEOSYS_PAL = 1
VIDEOSYS_720P60 = 4
VIDEOSYS_720P59DOT9 = 12
VIDEOSYS_720P50 = 5
VIDEOSYS_1080P30 = 20
VIDEOSYS_1080P29DOT9 = 28
VIDEOSYS_1080P25 = 21
VIDEOSYS_1080P24 = 36
VIDEOSYS_1080P23DOT9 = 44
VIDEOSYS_1080I30 = 68
VIDEOSYS_1080I29DOT9 = 76
VIDEOSYS_1080I25 = 69
VIDEOSYS_1080PSF30 = 52
VIDEOSYS_1080PSF29DOT9 = 60
VIDEOSYS_1080PSF25 = 53
VIDEOSYS_1080PSF24 = 84
VIDEOSYS_1080PSF23DOT9 = 92
NOTIFY_DEVICE_CHANGE = 1325
NOTIFY_BUS_RESET = 1326
BATTERY_MODE_DISABLE_CHARGING = 0
BATTERY_MODE_DISABLE_DISCHARGING_ARMING = 0
BATTERY_MODE_FORCE_ARMING = 0
BATTERY_MODE_ENABLE_PREVIEW_ARMING = 0
BATTERY_RUNTIME_1 = 1
BATTERY_RUNTIME_10 = 10
BATTERY_RUNTIME_60 = 60
BATTERY_RUNTIME_120 = 120
BATTERY_RUNTIME_180 = 180
BATTERY_RUNTIME_300 = 300
BATTERY_RUNTIME_600 = 600
BATTERY_NOT_PRESENT = 0
BATTERY_CHARGING = 1
BATTERY_CHARGING_HIGH = 2
BATTERY_CHARGED = 3
BATTERY_DISCHARGING = 4
BATTERY_LOW = 5
BATTERY_ARMED = 8
BATTERY_FAULT = 16
MAX_EXP_INDEX = 10
NVCR_CONT_REC = 0
NVCR_APV = 0
NVCR_REC_ONCE = 0
NVCR_NO_IMAGE = 0
FW_PHFW = 1
FW_FIRMWARE = 2
FW_FPGA = 3
FW_FLASH_FPGA = 4
FW_MAGAZINE = 5
FW_10G_FPGA = 6
FW_KERNEL_FLA = 7
FW_KERNEL_NFS = 8
ERR_Ok = 0
ERR_SimulatedCamera = 100
ERR_UnknownErrorCode = 101
ERR_BadResolution = 102
ERR_BadFrameRate = 103
ERR_BadPostTriggerFrames = 104
ERR_BadExposure = 105
ERR_BadEDRExposure = 106
ERR_BufferTooSmall = 107
ERR_CannotSetTime = 108
ERR_SerialNotFound = 109
ERR_CannotOpenStgFile = 110
ERR_UserInterrupt = 111
ERR_NoSimulatedImageFile = 112
ERR_SimulatedImageNot24bpp = 113
ERR_BadParam = 114
ERR_FlashCalibrationNewer = 115
ERR_ConnectedHeadsChanged = 117
ERR_NoHead = 118
ERR_NVMNotInstalled = 119
ERR_HeadNotAvailable = 120
ERR_FunctionNotAvailable = 121
ERR_Ph1394dllNotFound = 122
ERR_oldtNotFound = 123
ERR_BadFRPSteps = 124
ERR_BadFRPImgNr = 125
ERR_BadAutoExpLevel = 126
ERR_BadAutoExpRect = 127
ERR_BadDecimation = 128
ERR_BadCineParams = 129
ERR_IcmpNotAvailable = 130
ERR_CorrectResetLine = 131
ERR_CSRDoneInCamera = 132
ERR_ParamsChanged = 133
ERR_ParamReadOnly = 134
ERR_ParamWriteOnly = 135
ERR_ParamNotSupported = 136
ERR_IppWarning = 137
ERR_CannotOpenStpFile = 138
ERR_CannotSetStpParameters = 139
ERR_BadSensorMode = 140
ERR_CameraOffline = 141
ERR_InvalidFIle = 142
ERR_BatteryFailure = 175
ERR_PhStreamerdllNotFound = 180
ERR_CoreInit = 181
ERR_CoreErr = 182
ERR_DvrDeviceNotFound = 183
ERR_InvalidDevice = 184
ERR_XmlUrlAddress = 185
ERR_NULLPointer = -200
ERR_MemoryAllocation = -201
ERR_NoWindow = -202
ERR_CannotRegisterClient = -203
ERR_CannotUnregisterClient = -204
ERR_AsyncRead = -205
ERR_AsyncWrite = -206
ERR_IsochCIPHeader = -207
ERR_IsochDBCContinuity = -208
ERR_IsochNoHeader = -209
ERR_IsochAllocateResources = -210
ERR_IsochAttachBuffers = -211
ERR_IsochFreeResources = -212
ERR_IsochGetResult = -213
ERR_CannotReadTheSerialNumber = -214
ERR_SerialNumberOutOfRange = -215
ERR_UnknownCameraVersion = -216
ERR_GetImageTimeOut = -217
ERR_ImageNoOutOfRange = -218
ERR_CannotReadStgHeader = -220
ERR_ReadStg = -221
ERR_StgContents = -222
ERR_ReadStgOffsets = -223
ERR_ReadStgGains = -224
ERR_NotAStgFile = -225
ERR_StgSetupCheckSum = -226
ERR_StgSetup = -227
ERR_StgHardAdjCheckSum = -228
ERR_StgHardAdj = -229
ERR_StgDifferentSerials = -230
ERR_WriteStg = -231
ERR_NoCine = -232
ERR_CannotOpenDevice = -233
ERR_TimeBufferSize = -234
ERR_CannotWriteCineParams = -236
ERR_NVMError = -250
ERR_NoNVM = -251
ERR_FlashEraseTimeout = -253
ERR_FlashWriteTimeout = -254
ERR_FlashContents = -255
ERR_FlashOffsetsCheckSum = -256
ERR_FlashGainsCheckSum = -257
ERR_TooManyCameras = -258
ERR_NoResponseFromCamera = -259
ERR_MessageFromCamera = -260
ERR_BadImgResponse = -261
ERR_AllPixelsBad = -262
ERR_BadTimeResponse = -263
ERR_GetTimeTimeOut = -264
ERR_BadAudioResponse = -265
ERR_GetAudioTimeOut = -266
ERR_InBlockTooBig = -270
ERR_OutBufferTooSmall = -271
ERR_BlockNotValid = -272
ERR_DataAfterPadding = -273
ERR_InvalidSlash = -274
ERR_UnknownChar = -275
ERR_MalformedLine = -276
ERR_EndMarkerNotFound = -277
ERR_NoTimeData = -280
ERR_NoExposureData = -281
ERR_NoRangeData = -282
ERR_NotIncreasingTime = -283
ERR_BadTriggerTime = -284
ERR_TimeOut = -285
ERR_NullWeightsSum = -286
ERR_BadCount = -287
ERR_CannotChangeRecordedCine = -288
ERR_BadSliceCount = -289
ERR_NotAvailable = -290
ERR_BadImageInterval = -291
ERR_BadCameraNumber = -292
ERR_BadCineNumber = -293
ERR_BadSyncObject = -294
ERR_IcmpEchoError = -295
ERR_MlairReadFirstPacket = -296
ERR_MlairReadPacket = -297
ERR_MlairIncorrectOrder = -298
ERR_MlairStartRecorder = -299
ERR_MlairStopRecorder = -300
ERR_MlairOpenFile = -301
ERR_CmdMutexTimeOut = -302
ERR_CmdMutexAbandoned = -303
ERR_UnsupportedConversion = -304
ERR_TenGigLostPacket = -305
ERR_TooManyImgReq = -306
ERR_BadImRange = -307
ERR_ImgBufferTooSmall = -308
ERR_ImgSize0 = -309
ERR_IppError = -310
ERR_pscpError = -311
ERR_plinkError = -312
ERR_phFWDecodeError = -313
ERR_phFolderError = -314
ERR_CameraUpDateError = -315
ERR_CineMagBusy = -320
ERR_UnknownToken = -321
ERR_BadPortNumber = -330
ERR_PortNotProg = -331
ERR_BadSigNumber = -332
ERR_EarlyTrigger = -200281
ERR_NoStrobePulses = -200284
GI_INTERPOLATED = 4
GI_ONEHEAD = 8
GI_HEADMASK = 0
GI_BPP16 = 0
GI_PACKED = 0
GI_PACKED12L = 0
GI_DENORMALIZED = 0


# datatypes defined in header files
# phint.h
# PhCon.h
# PhFile.h
class tagRECT(ct.Structure):
    _fields_ = [
        ("left", ct.c_int32),
        ("top",  ct.c_int32),
        ("right", ct.c_int32),
        ("bottom", ct.c_int32)
    ]

class tagTIME64(ct.Structure):
    _fields_ = [
        ("fractions", ct.c_uint32),
        ("seconds", ct.c_uint32)
    ]

class tagACQUIPARAMS(ct.Structure):
    _fields_ = [
        ("ImWidth", ct.c_uint),
        ("ImHeight", ct.c_uint),
        ("FrameRateInt", ct.c_uint),
        ("Exposure", ct.c_uint),
        ("EDRExposure", ct.c_uint),
        ("ImDelay", ct.c_uint),
        ("PTFrames", ct.c_uint),
        ("ImCount", ct.c_uint),
        ("SyncImaging", ct.c_uint),
        ("AutoExposure", ct.c_uint),
        ("AutoExpLevel", ct.c_uint),
        ("AutoExpSpeed", ct.c_uint),
        ("AutoExpRect", tagRECT),
        ("Recorded", ct.c_bool),
        ("TriggerTime", tagTIME64),
        ("RecordedCount", ct.c_uint),
        ("FirstIm", ct.c_int),
        ("FRPSteps", ct.c_uint),
        ("FRPImgNr", ct.c_int * 16),
        ("FRPRate", ct.c_uint * 16),
        ("FRPExp", ct.c_uint * 16),
        ("Decimation", ct.c_uint),
        ("BitDepth", ct.c_uint),
        ("CamGainRed", ct.c_uint),
        ("CamGainGreen", ct.c_uint),
        ("CamGainBlue", ct.c_uint),
        ("CamGain", ct.c_uint),
        ("ShutterOff", ct.c_bool),
        ("CFA", ct.c_uint),
        ("CineName", ct.c_char * 256),
        ("Description", ct.c_char * 4096),
        ("FRPShape", ct.c_uint * 16),
        ("dFrameRate", ct.c_double)
    ]

class tagCINESTATUS(ct.Structure):
    _fields_ = [
        ("Stored", ct.c_bool),
        ("Active", ct.c_bool),
        ("Triggered", ct.c_bool)
    ]


class tagIMRANGE(ct.Structure):
    _fields_ = [
        ("First", ct.c_int),
        ("Cnt", ct.c_uint)
    ]

class tagIH(ct.Structure):
    _fields_ = [
        ("biSize", ct.c_uint),
        ("biWidth", ct.c_long),
        ("biHeight", ct.c_long),
        ("biPlanes", ct.c_ushort),
        ("biBitCount", ct.c_ushort),
        ("biCompression", ct.c_uint),
        ("biSizeImage", ct.c_ushort),
        ("biXPelsPerMeter", ct.c_long),
        ("biYPelsPerMeter", ct.c_long),
        ("biClrUsed", ct.c_uint),
        ("biClrImportant", ct.c_uint),
        ("BlackLevel", ct.c_int),
        ("WhiteLevel", ct.c_int)
    ]

def struct2dict(struct):
    d = dict((field, getattr(struct, field)) for field, _ in struct._fields_)

    for k, v in d.items():
        if hasattr(v, "_length_") and hasattr(v, "_type_"):
            d[k] = list(v) # array type
        elif isinstance(v, ct.Structure) and hasattr(v, "_fields_"):
            d[k] = struct2dict(v) # another structure

    return d


class camera:
    def __init__(self):
        pass


class phantom_cam(camera):
    _phcon = None
    _phint = None
    _phfile = None

    def __init__(self, root_dir=Path(r"C:\Users\q2ilab\Documents\Phantom\PhSDK800\Bin\Win64"), initialize=True):

        self.initialized = initialize
        if self.initialize:
            # load dll's we will need
            self._phcon = ct.cdll.LoadLibrary(str(root_dir / "PhCon.Dll"))
            self._phint = ct.cdll.LoadLibrary(str(root_dir / "PhInt.Dll"))
            self._phfile = ct.cdll.LoadLibrary(str(root_dir / "PhFile.Dll"))


            getattr(self._phcon, "PhLVRegisterClientEx")(None, None, ct.c_int(800))

            # see if cameras connected
            camCount = ct.c_int()
            getattr(self._phcon, "PhGetCameraCount")(ct.byref(camCount))

            # get camera info
            serial = ct.c_int()

            sbuff = ct.create_string_buffer(256)
            name = ct.c_char_p(ct.addressof(sbuff))

            getattr(self._phcon, "PhGetCameraID")(ct.c_int(0), ct.byref(serial), name)
            self.serial_num = serial.value
            self.name = name.value.decode("ascii")

            self.cam_index = ct.c_int(0)

            # get max number of cines
            self.max_cine_ct = getattr(self._phcon, "PhMaxCineCnt")(self.cam_index)


    def initialize(self, **kwargs):
        self.__init__(initialize=True, **kwargs)

    def __del__(self):
        if self.initialized:
            getattr(self._phcon, "PhLVUnregisterClient")()

    def get_error(self, code):
        msg = ct.create_string_buffer(256)
        getattr(self._phcon, "PhGetErrorMessage")(code, ct.byref(msg))

        return msg.value

    def software_trigger(self):
        """
        send software trigger
        """
        getattr(self._phcon, "PhSendSoftwareTrigger")(self.cam_index)

    def get_params(self, cine_no=1):
        """
        Convert the output to a dictionary using struct2dict
        """
        aqParams = tagACQUIPARAMS()
        getattr(self._phcon, "PhGetCineParams")(self.cam_index, ct.c_int(cine_no), ct.byref(aqParams), None)

        return aqParams

    def quiet_fan(self, state: bool):
        state_c = ct.c_int(state)
        result = getattr(self._phcon, "PhSet")(self.cam_index, ct.c_uint(gsQuiet), ct.byref(state_c))
        if result != 0:
            raise Exception(self.get_error(result))

    # deal with cines
    def set_cines(self, ncins):
        """
        Set number of cines and percent of memory each occupies
        """
        percents = [100 // ncins] * ncins

        # if nimgs is None:
        #     percents = [100 // ncins] * ncins
        # else:
        #     ncins += 1
        #
            # # todo: grab max number of frames with sdk
            # max_frames = 40260
            # n_leftover = max_frames - np.sum(nimgs)
            #
            # percents = [n for n in nimgs] + [n_leftover]

        num_parts = ct.c_int(ncins)
        pweights = (ct.c_uint * ncins)(*percents) # percent of memory each cine occupies
        result = getattr(self._phcon, "PhSetPartitions")(self.cam_index, num_parts, ct.byref(pweights))
        if result != 0:
            raise Exception(self.get_error(result))

    def get_cine_status(self):
        """
        Get cine status
        """

        cine_status = (tagCINESTATUS * self.max_cine_ct)()
        result = getattr(self._phcon, "PhGetCineStatus")(self.cam_index, ct.byref(cine_status))
        if result != 0:
            raise Exception(self.get_error(result))

        return cine_status

    def record_cine(self, cine_no=None):
        """
        Record cine
        """

        if cine_no is None:
            result = getattr(self._phcon, "PhRecordCine")(self.cam_index)
        else:
            if cine_no < 0:
                raise ValueError("cine_no must be > 0")

            result = getattr(self._phcon, "PhRecordSpecificCine")(self.cam_index, ct.c_int(cine_no))

        if result != 0:
            raise Exception(self.get_error(result))

    def save_cine(self, cine_no, fname, first_image=None, img_count=None):

        ch = ct.pointer(ct.c_int())
        result = getattr(self._phfile, "PhNewCineFromCamera")(self.cam_index, ct.c_int(cine_no), ct.byref(ch))
        if result != 0:
            raise Exception(self.get_error(result))

        # first image
        if first_image is None:
            img_no = ct.c_int()
            result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_FIRSTIMAGENO), ct.byref(img_no))
            if result != 0:
                raise Exception(self.get_error(result))
        else:
            img_no = ct.c_int(first_image)

        # image count
        if img_count is None:
            img_ct = ct.c_uint()
            result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_IMAGECOUNT), ct.byref(img_ct))
            if result != 0:
                raise Exception(self.get_error(result))
        else:
            img_ct = ct.c_uint(img_count)

        # set range to save
        imgRng = tagIMRANGE()
        imgRng.First = img_no
        imgRng.Cnt = img_ct

        result = getattr(self._phfile, "PhSetCineInfo")(ch, ct.c_uint(GCI_SAVERANGE), ct.byref(imgRng))
        if result != 0:
            raise Exception(self.get_error(result))

        # imgRngGet = tagIMRANGE()
        # result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_uint(GCI_SAVERANGE), ct.byref(imgRngGet))
        # print(imgRngGet.First)
        # print(imgRngGet.Cnt)

        # set 12L packed
        pack_type = ct.c_uint(2)
        result = getattr(self._phfile, "PhSetCineInfo")(ch, ct.c_uint(GCI_SAVEPACKED), ct.byref(pack_type))
        if result != 0:
            raise Exception(self.get_error(result))

        # set save file type
        file_type = ct.c_int(SIFILE_TIF12)
        result = getattr(self._phfile, "PhSetCineInfo")(ch, ct.c_uint(GCI_SAVEFILETYPE), ct.byref(file_type))
        if result != 0:
            raise Exception(self.get_error(result))

        # set save fname
        fname_c = ct.create_string_buffer(bytes(str(fname), "ascii"))
        result = getattr(self._phfile, "PhSetCineInfo")(ch, ct.c_uint(GCI_SAVEFILENAME), ct.byref(fname_c))
        if result != 0:
            raise Exception(self.get_error(result))

        # fname_cget = ct.create_string_buffer(256)
        # result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_uint(GCI_SAVEFILENAME), ct.byref(fname_cget))
        # print(fname_cget.value)

        # save
        result = getattr(self._phfile, "PhWriteCineFile")(ch, None)
        if result != 0:
            raise Exception(self.get_error(result))

    def destroy_cine(self, cine_num):
        """
        Destroy a specific cine
        """

        ch = ct.pointer(ct.c_int())
        result = getattr(self._phfile, "PhNewCineFromCamera")(self.cam_index, cine_num, ct.byref(ch))
        if result != 0:
            raise Exception(self.get_error(result))

        results = getattr(self._phfile, "PhDestroyCine")(ch)
        if result != 0:
            raise Exception(self.get_error(result))

    def getImage(self):
        """
        Read image from live Cine
        @return:
        """

        ch = ct.pointer(ct.c_int())
        result = getattr(self._phfile, "PhGetCineLive")(self.cam_index, ct.byref(ch))
        if result != 0:
            raise Exception(self.get_error(result))

        buffer_size = ct.c_int()
        result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_MAXIMGSIZE), ct.byref(buffer_size))
        if result != 0:
            raise Exception(self.get_error(result))

        pixel = ct.create_string_buffer(buffer_size.value)
        img_header = tagIH()

        result = getattr(self._phfile, "PhGetCineImage")(ch, None, ct.byref(pixel), buffer_size,
                                                      ct.byref(img_header))
        if result != 0:
            raise Exception(self.get_error(result))

        # seems this always returns an RGB image. We will only grab the first image
        # nimgs = len(pixel) // (2 * ny * nx)
        # ntrim = len(pixel) // 2 - nimgs * ny * nx
        # img = np.frombuffer(pixel, dtype=np.uint16)[:-ntrim].reshape([nimgs, ny, nx])

        ny = img_header.biHeight
        nx = img_header.biWidth
        img = np.frombuffer(pixel[: ny * nx * 2], dtype=np.uint16).reshape([ny, nx])

        return img

    def get_recorded_img(self, cine_no=1, img_start=0, img_end=None):
        if cine_no < 1:
            raise ValueError(f"cine_no must be >= 1 but was {cine_no:d}")

        if img_end is None:
            img_end = img_start + 1

        cine_num = ct.c_int(cine_no)  # start at 1

        # get cine handle
        ch = ct.pointer(ct.c_int())
        result = getattr(self._phfile, "PhNewCineFromCamera")(self.cam_index, cine_num, ct.byref(ch))
        if result != 0:
            raise Exception(self.get_error(result))

        # if handle not yet initialized...
        if result != 0:
            return

        # get buffer size
        buffer_size = ct.c_int()
        result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_MAXIMGSIZE), ct.byref(buffer_size))
        if result != 0:
            raise Exception(self.get_error(result))

        # first image
        # img_no = ct.c_int()
        # result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_FIRSTIMAGENO), ct.byref(img_no))
        # if result != 0:
        #     raise Exception(self.get_error(result))

        # image count
        img_ct = ct.c_uint()
        result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_IMAGECOUNT), ct.byref(img_ct))
        if result != 0:
            raise Exception(self.get_error(result))

        # image sizes
        width = ct.c_int()
        result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_IMWIDTH), ct.byref(width))
        if result != 0:
            raise Exception(self.get_error(result))

        height = ct.c_int()
        result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_IMHEIGHT), ct.byref(height))
        if result != 0:
            raise Exception(self.get_error(result))
        nimgs = img_end - img_start
        ny = height.value
        nx = width.value
        img = np.empty((nimgs, ny, nx), dtype=np.uint16)

        for ii in range(nimgs):

            # get one image
            imgRng = tagIMRANGE()
            imgRng.First = ct.c_int(img_start + ii)
            imgRng.Cnt = ct.c_uint(1)

            # # get all images
            # always got buffer error when tried to do this for more than one image
            # imgRng = tagIMRANGE()
            # imgRng.First = img_no
            # imgRng.Cnt = ct.c_uint(2)

            pixel = ct.create_string_buffer(buffer_size.value)
            img_header = tagIH()

            result = getattr(self._phfile, "PhGetCineImage")(ch, ct.byref(imgRng), ct.byref(pixel), buffer_size, ct.byref(img_header))
            # print(self.get_error(result))

            # why are there an extra 40 bytes on the end?
            img[ii] = np.frombuffer(pixel, dtype=np.uint16)[:-20].reshape([ny, nx])

        return img

    def stopSequenceAcquisition(self):
        self.record_cine(0)  # stop recording

    def startContinuousSequenceAcquisition(self, *args):
        pass

    def getImageWidth(self):
        ch = ct.pointer(ct.c_int())
        result = getattr(self._phfile, "PhGetCineLive")(self.cam_index, ct.byref(ch))
        if result != 0:
            raise Exception(self.get_error(result))

        width = ct.c_int()
        result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_IMWIDTH), ct.byref(width))
        if result != 0:
            raise Exception(self.get_error(result))

        return width.value

    def getImageHeight(self):
        ch = ct.pointer(ct.c_int())
        result = getattr(self._phfile, "PhGetCineLive")(self.cam_index, ct.byref(ch))
        if result != 0:
            raise Exception(self.get_error(result))

        height = ct.c_int()
        result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_IMHEIGHT), ct.byref(height))
        if result != 0:
            raise Exception(self.get_error(result))

        return height.value

    def getExposure(self):
        """
        Exposure time in ms
        @return:
        """
        ch = ct.pointer(ct.c_int())
        result = getattr(self._phfile, "PhGetCineLive")(self.cam_index, ct.byref(ch))
        if result != 0:
            raise Exception(self.get_error(result))

        exposure = ct.c_uint()
        result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_EXPOSURE), ct.byref(exposure))
        if result != 0:
            raise Exception(self.get_error(result))

        return (exposure.value / 1e3)

    def setAcqParams(self, cine_no, **kwargs):
        """
        Set acquisition parameters. Give as keyword arguments. Any acquisition parameters not provided will keep
        the same value as before
        """
        cn = ct.c_int(cine_no)

        # initialize form existing parameters
        params = self.get_params(cine_no)

        for fn, dt in params._fields_:
            if fn in kwargs:
                setattr(params, fn, kwargs[fn])

        result = getattr(self._phcon, "PhSetCineParams")(self.cam_index, cn, ct.byref(params))
        if result != 0:
            raise Exception(self.get_error(result))
        # paramsa are input and will also contain actual output values

        return params

    def snapImage(self):
        pass

    def getLastImage(self):
        return self.getImage()

    def getCameraDevice(self):
        return self.name

    def setExposure(self):
        pass

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import time

    c = phantom_cam()


    # result = getattr(c._phcon, "PhRecordCine")(c.cam_index)
    # print(c.get_error(result))
    # c.trigger()
    # time.sleep(3000)
