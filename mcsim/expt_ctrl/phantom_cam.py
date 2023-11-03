"""
Control Phantom camera using their SDK and ctypes
"""

import numpy as np
import ctypes as ct
from pathlib import Path
import struct
import dask
import dask.array as da
from typing import Optional

# ###############################
# Codes defined in PhFile.h
# note: generated these programmatically
# ###############################
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

# ###############################
# codes defined in PhCon.h
# ###############################
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

# ###############################
# datatypes defined in header files
# phint.h
# PhCon.h
# PhFile.h
# ###############################
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

class PhantomException(Exception):
    pass

# ###############################
# phantom camera
# ###############################
class phantom_cam(camera):
    _phcon = None
    _phint = None
    _phfile = None
    width = 1280
    height = 960

    def __init__(self,
                 root_dir=Path(r"C:\Users\q2ilab\Documents\Phantom\PhSDK800\Bin\Win64"),
                 initialize=True):
        """

        @param root_dir: Directory containing PhCon.Dll, PhInt.Dll, and PhFile.Dll
        @param initialize:
        """

        super().__init__()
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
            # todo: implement logic for this
            self.cine_running = False

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

    def get_params(self, cine_no: int = 1):
        """
        The user may convert the output to a dictionary using struct2dict
        """
        aqParams = tagACQUIPARAMS()
        getattr(self._phcon, "PhGetCineParams")(self.cam_index, ct.c_int(cine_no), ct.byref(aqParams), None)

        return aqParams

    def quiet_fan(self, state: bool):
        state_c = ct.c_int(state)
        result = getattr(self._phcon, "PhSet")(self.cam_index, ct.c_uint(gsQuiet), ct.byref(state_c))
        if result != 0:
            raise PhantomException(self.get_error(result))
        
    def set_black_reference(self):
        result = getattr(self._phcon, "PhBlackReferenceCI")(self.cam_index, None)
        if result != 0:
            raise PhantomException(self.get_error(result))

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
            raise PhantomException(self.get_error(result))

    def get_cine_status(self):
        """
        Get cine status
        """

        cine_status = (tagCINESTATUS * self.max_cine_ct)()
        result = getattr(self._phcon, "PhGetCineStatus")(self.cam_index, ct.byref(cine_status))
        if result != 0:
            raise PhantomException(self.get_error(result))

        return cine_status

    def record_cine(self, cine_no=None):
        """
        Record cine
        """

        if cine_no is None:
            result = getattr(self._phcon, "PhRecordCine")(self.cam_index)
        else:
            if cine_no < 0:
                raise ValueError("cine_no must be >= 0")

            result = getattr(self._phcon, "PhRecordSpecificCine")(self.cam_index, ct.c_int(cine_no))

        if result != 0:
            raise PhantomException(self.get_error(result))

    def save_cine(self,
                  cine_no: int,
                  fname: str,
                  first_image=None,
                  img_count=None,
                  file_type: str = "cine raw"):
        """

        :param cine_no:
        :param fname:
        :param first_image:
        :param img_count:
        :param file_type: "cine raw", "cine", "tif16", "tif12", "tif8", among other options
        :return:
        """

        file_types = {"tif16": SIFILE_TIF16,
                      "tif12": SIFILE_TIF12,
                      "tif8": SIFILE_TIF8,
                      # "raw": ct.c_int(SIFILE_RAW),
                      "dng": SIFILE_DNG,
                      "dpx": SIFILE_DPX,
                      "cine raw": MIFILE_RAWCINE,
                      "cine": MIFILE_CINE,
                      "JPEG cine": MIFILE_JPEGCINE,
                      "AVI": MIFILE_AVI,
                      "multipage TIFF": MIFILE_TIFCINE,
                      "MPEG": MIFILE_MPEG,
                      "QTIME": MIFILE_QTIME,
                      "MP4": MIFILE_MP4,
                      }

        if file_type not in file_types.keys():
            raise ValueError(f"file type {file_type:s} is not supported")

        ch = ct.pointer(ct.c_int())
        result = getattr(self._phfile, "PhNewCineFromCamera")(self.cam_index, ct.c_int(cine_no), ct.byref(ch))
        if result != 0:
            raise PhantomException(self.get_error(result))

        # first image
        if first_image is None:
            img_no = ct.c_int()
            result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_FIRSTIMAGENO), ct.byref(img_no))
            if result != 0:
                raise PhantomException(self.get_error(result))
        else:
            img_no = ct.c_int(first_image)

        # image count
        if img_count is None:
            img_ct = ct.c_uint()
            result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_IMAGECOUNT), ct.byref(img_ct))
            if result != 0:
                raise PhantomException(self.get_error(result))
        else:
            img_ct = ct.c_uint(img_count)

        # set range to save
        imgRng = tagIMRANGE()
        imgRng.First = img_no
        imgRng.Cnt = img_ct

        result = getattr(self._phfile, "PhSetCineInfo")(ch, ct.c_uint(GCI_SAVERANGE), ct.byref(imgRng))
        if result != 0:
            raise PhantomException(self.get_error(result))

        # imgRngGet = tagIMRANGE()
        # result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_uint(GCI_SAVERANGE), ct.byref(imgRngGet))
        # print(imgRngGet.First)
        # print(imgRngGet.Cnt)

        # set 12L packed
        pack_type = ct.c_uint(2)
        result = getattr(self._phfile, "PhSetCineInfo")(ch, ct.c_uint(GCI_SAVEPACKED), ct.byref(pack_type))
        if result != 0:
            raise PhantomException(self.get_error(result))

        # set save file type
        # file_type = ct.c_int(SIFILE_TIF12)
        file_type_int = ct.c_int(file_types[file_type])
        result = getattr(self._phfile, "PhSetCineInfo")(ch, ct.c_uint(GCI_SAVEFILETYPE), ct.byref(file_type_int))
        if result != 0:
            raise PhantomException(self.get_error(result))

        # set save fname
        fname_c = ct.create_string_buffer(bytes(str(fname), "ascii"))
        result = getattr(self._phfile, "PhSetCineInfo")(ch, ct.c_uint(GCI_SAVEFILENAME), ct.byref(fname_c))
        if result != 0:
            raise PhantomException(self.get_error(result))

        # fname_cget = ct.create_string_buffer(256)
        # result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_uint(GCI_SAVEFILENAME), ct.byref(fname_cget))
        # print(fname_cget.value)

        # save
        result = getattr(self._phfile, "PhWriteCineFile")(ch, None)
        if result != 0:
            raise PhantomException(self.get_error(result))

    def destroy_cine(self, cine_num):
        """
        Destroy a specific cine
        """

        ch = ct.pointer(ct.c_int())
        result = getattr(self._phfile, "PhNewCineFromCamera")(self.cam_index, cine_num, ct.byref(ch))
        if result != 0:
            raise PhantomException(self.get_error(result))

        results = getattr(self._phfile, "PhDestroyCine")(ch)
        if result != 0:
            raise PhantomException(self.get_error(result))

    def getImage(self):
        """
        Read image from live Cine
        @return:
        """

        ch = ct.pointer(ct.c_int())
        result = getattr(self._phfile, "PhGetCineLive")(self.cam_index, ct.byref(ch))
        if result != 0:
            raise PhantomException(self.get_error(result))

        buffer_size = ct.c_int()
        result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_MAXIMGSIZE), ct.byref(buffer_size))
        if result != 0:
            raise PhantomException(self.get_error(result))

        pixel = ct.create_string_buffer(buffer_size.value)
        img_header = tagIH()

        result = getattr(self._phfile, "PhGetCineImage")(ch, None, ct.byref(pixel), buffer_size,
                                                      ct.byref(img_header))
        if result != 0:
            raise PhantomException(self.get_error(result))

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
            raise PhantomException(self.get_error(result))

        # if handle not yet initialized...
        if result != 0:
            return

        # get buffer size
        buffer_size = ct.c_int()
        result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_MAXIMGSIZE), ct.byref(buffer_size))
        if result != 0:
            raise PhantomException(self.get_error(result))

        # first image
        # img_no = ct.c_int()
        # result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_FIRSTIMAGENO), ct.byref(img_no))
        # if result != 0:
        #     raise PhantomException(self.get_error(result))

        # image count
        img_ct = ct.c_uint()
        result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_IMAGECOUNT), ct.byref(img_ct))
        if result != 0:
            raise PhantomException(self.get_error(result))

        # image sizes
        width = ct.c_int()
        result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_IMWIDTH), ct.byref(width))
        if result != 0:
            raise PhantomException(self.get_error(result))

        height = ct.c_int()
        result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_IMHEIGHT), ct.byref(height))
        if result != 0:
            raise PhantomException(self.get_error(result))
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

    def isSequenceRunning(self):
        # todo: what best to do here?
        # return self.cine_running
        return False

    def stopSequenceAcquisition(self):
        self.cine_running = False
        # todo: shouldn't this be 1 not zero?
        self.record_cine(0)  # stop recording

    def startContinuousSequenceAcquisition(self, exposure_ms: float = 0.):

        try:
            self.setExposure(exposure_ms)
        except Exception as e:
            print(e)

    def getImageWidth(self):
        ch = ct.pointer(ct.c_int())
        result = getattr(self._phfile, "PhGetCineLive")(self.cam_index, ct.byref(ch))
        if result != 0:
            raise PhantomException(self.get_error(result))

        width = ct.c_int()
        result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_IMWIDTH), ct.byref(width))
        if result != 0:
            raise PhantomException(self.get_error(result))

        return width.value

    def getImageHeight(self):
        ch = ct.pointer(ct.c_int())
        result = getattr(self._phfile, "PhGetCineLive")(self.cam_index, ct.byref(ch))
        if result != 0:
            raise PhantomException(self.get_error(result))

        height = ct.c_int()
        result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_IMHEIGHT), ct.byref(height))
        if result != 0:
            raise PhantomException(self.get_error(result))

        return height.value

    def getExposure(self):
        """
        Exposure time in ms
        @return:
        """
        ch = ct.pointer(ct.c_int())
        result = getattr(self._phfile, "PhGetCineLive")(self.cam_index, ct.byref(ch))
        if result != 0:
            raise PhantomException(self.get_error(result))

        exposure = ct.c_uint()
        result = getattr(self._phfile, "PhGetCineInfo")(ch, ct.c_int(GCI_EXPOSURE), ct.byref(exposure))
        if result != 0:
            raise PhantomException(self.get_error(result))

        return (exposure.value / 1e3)

    def setExposure(self, exposure_ms: float, cine_no: int = 0):
        """
        Set camera exposure time
        """
        exposure_ns = int(np.round(exposure_ms * 1e6))

        # older parameters like exposure are not set using the PhSetCineInfo function, but instead by setting
        # the acquisition parameters sturcture

        acq_params = self.setAcqParams(cine_no=cine_no, Exposure=exposure_ns)
        exposure_set_ms = acq_params.Exposure / 1e6

        return exposure_set_ms

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
            raise PhantomException(self.get_error(result))
        # params are input and will also contain actual output values

        return params

    def getROI(self):
        # todo: generalize
        nx = self.getImageWidth()
        ny = self.getImageHeight()

        nx_start = (self.width - nx) // 2
        ny_start = (self.height - ny) // 2

        return [nx_start, ny_start, nx, ny]

    def setROI(self):
        raise NotImplementedError()

    def snapImage(self):
        pass

    def getLastImage(self):
        return self.getImage()

    def getCameraDevice(self):
        return self.name


# ###############################
# load Cine
# ###############################
def imread_cine(fname: str,
                start_index: int = 0,
                end_index: Optional[int] = None,
                read_setup_info: bool = False) -> (dask.array.core.Array, dict):
    """
    Read images saved in a grayscale Cine file using 12L packing

    :param fname: File path
    :param start_index:
    :param end_index:
    :param read_setup_info:
    :return imgs, metadata: dask array of images
    """

    # first open file and read metadata
    # mainly, need to grab offsets that tell us where the images are stored
    with open(fname, "rb") as f:
        # ########################
        # read Cine file header
        # ########################
        file_type = f.read(2).decode()
        headersize = struct.unpack("<H", f.read(2))[0]

        # read rest of info
        header_bytes = f.read(headersize - 4)
        header_info = struct.unpack("HHiIiIIIIi", header_bytes[:36])
        header_names = ["compression",
                        "version",
                        "first_movie_image",
                        "total_image_count",
                        "first_image_no",
                        "image_count",
                        "off_image_header",
                        "off_setup",
                        "off_image_offsets",
                        "trigger_time",
                        ]

        # info as dictionary
        header = {"filetype": file_type,
                  "headersize": headersize}
        header.update(dict(zip(header_names, header_info)))

        # ########################
        # read image offsets
        # ########################
        f.seek(header["off_image_offsets"])
        # pointers to image are 64 bits
        offsets = np.frombuffer(f.read(header["image_count"] * 8), np.uint64)

        # ########################
        # read bit map info header
        # ########################
        f.seek(header["off_image_header"])

        bit_map_i = struct.unpack("IiiHHIIiiII", f.read(40))
        names = ["bi_size",
                 "bi_width",
                 "bi_height",
                 "bi_planes",
                 "bi_bit_count",
                 "bi_compression",
                 "bi_size_image",
                 "bi_x_pels_per_meter",
                 "bi_y_pels_per_meter",
                 "bi_clr_used",
                 "bi_clr_important"]

        bit_map_info = dict(zip(names, bit_map_i))

        # ########################n
        # read setup information
        # ########################
        setup_info = {}

        if read_setup_info:
            f.seek(header["off_setup"])

            max_len_description_old = 121  # defined in
            old_max_file_name = 65
            max_len_description = 4096
            max_std_str_sz = 256  # defined in phint.h

            # todo: store in a dictionary
            # setup_bytes = f.read(10412)

            # setup_i = list(struct.unpack("HHHHHHHBBBBB", setup_bytes))
            #
            # description_old_bytes = struct.unpack(max_len_description_old * "c", f.read(max_len_description_old))
            # setup_i.append("".join(c.decode() for c in description_old_bytes if c != b"\x00"))
            #
            # setup_i += struct.unpack("", f.read())

            setup_names = ["frame_rate",
                           "shutter",
                           "post_trigger",
                           "frame_delay",
                           "aspect_ratio",
                           "res7",
                           "res8",
                           "res9",
                           "res10",
                           "res11",
                           "trig_frame",
                           "res12",
                           "description_old_bytes"
                           ]

            setup_info["frame_rate"] = struct.unpack("<H", f.read(2))[0]
            setup_info["shutter"] = struct.unpack("<H", f.read(2))[0]
            setup_info["post_trigger"] = struct.unpack("<H", f.read(2))[0]
            setup_info["frame_delay"] = struct.unpack("<H", f.read(2))[0]
            setup_info["aspect_ratio"] = struct.unpack("<H", f.read(2))[0]
            setup_info["res7"] = struct.unpack("<H", f.read(2))[0]
            setup_info["res8"] = struct.unpack("<H", f.read(2))[0]
            setup_info["res9"] = struct.unpack("B", f.read(1))[0]
            setup_info["res10"] = struct.unpack("B", f.read(1))[0]
            setup_info["res11"] = struct.unpack("B", f.read(1))[0]
            setup_info["trig_frame"] = struct.unpack("B", f.read(1))[0]
            setup_info["res12"] = struct.unpack("B", f.read(1))[0]

            description_old_bytes = struct.unpack(max_len_description_old * "c", f.read(max_len_description_old))
            setup_info["description_old_bytes"] = "".join(c.decode() for c in description_old_bytes if c != b"\x00")

            setup_info["mark"] = struct.unpack("<H", f.read(2))[0]
            setup_info["length"] = struct.unpack("<H", f.read(2))[0]
            setup_info["res13"] = struct.unpack("<H", f.read(2))[0]
            setup_info["sig_option"] = struct.unpack("<H", f.read(2))[0]
            setup_info["bin_channels"] = struct.unpack("<h", f.read(2))[0]
            setup_info["samples_per_image"] = struct.unpack("B", f.read(1))[0]

            # setup_info["bin_names"] = struct.unpack(8 * 11 * "c", f.read(8 * 11))
            bin_names = struct.unpack(8 * 11 * "c", f.read(8 * 11))
            setup_info["bin_names"] = "".join(c.decode() for c in bin_names if c != b"\x00")

            setup_info["ana_option"] = struct.unpack("<H", f.read(2))[0]
            setup_info["ana_channels"] = struct.unpack("<h", f.read(2))[0]
            setup_info["res6"] = struct.unpack("<B", f.read(1))[0]
            setup_info["ana_board"] = struct.unpack("<B", f.read(1))[0]
            setup_info["ch_option"] = struct.unpack("H" * 8, f.read(2 * 8))
            setup_info["ana_gain"] = struct.unpack("f" * 8, f.read(4 * 8))


            # setup_info["ana_unit"] = struct.unpack("c" * 8 * 6, f.read(8 * 6))
            ana_unit = struct.unpack("c" * 8 * 6, f.read(8 * 6))
            setup_info["ana_unit"] = "".join(c.decode() for c in ana_unit if c != b"\x00")

            # setup_info["ana_name"] = struct.unpack("c" * 8 * 11, f.read(8 * 11))
            ana_name = struct.unpack("c" * 8 * 11, f.read(8 * 11))
            setup_info["ana_name"] = "".join(c.decode() for c in ana_name if c != b"\x00")

            setup_info["lfirst_image"] = struct.unpack("<i", f.read(4))[0]
            setup_info["dw_image_count"] = struct.unpack("<I", f.read(4))[0]
            setup_info["nq_factor"] = struct.unpack("<h", f.read(2))[0]
            setup_info["w_cine_file_type"] = struct.unpack("<H", f.read(2))[0]

            sz_cine_path_bytes = struct.unpack("c" * 4 * old_max_file_name, f.read(4 * old_max_file_name))
            setup_info["sz_cine_path"] = "".join(c.decode() for c in sz_cine_path_bytes if c != b"\x00")

            setup_info["res14"] = struct.unpack("<H", f.read(2))[0]
            setup_info["res15"] = struct.unpack("B", f.read(1))[0]
            setup_info["res16"] = struct.unpack("B", f.read(1))[0]
            setup_info["res17"] = struct.unpack("<H", f.read(2))[0]
            setup_info["res18"] = struct.unpack("d", f.read(8))[0]
            setup_info["res19"] = struct.unpack("d", f.read(8))[0]
            setup_info["res20"] = struct.unpack("<H", f.read(2))[0]
            setup_info["res1"] = struct.unpack("<i", f.read(4))[0]
            setup_info["res2"] = struct.unpack("<i", f.read(4))[0]
            setup_info["res3"] = struct.unpack("<i", f.read(4))[0]
            setup_info["im_width"] = struct.unpack("<H", f.read(2))[0]
            setup_info["im_height"] = struct.unpack("<H", f.read(2))[0]
            setup_info["edr_shutter_16"] = struct.unpack("<H", f.read(2))[0]
            setup_info["serial"] = struct.unpack("<I", f.read(4))[0]
            setup_info["saturation"] = struct.unpack("<i", f.read(4))[0]
            setup_info["res5"] = struct.unpack("B", f.read(1))[0]
            setup_info["auto_exposure"] = struct.unpack("<I", f.read(4))[0]
            setup_info["bfliph"] = struct.unpack("<i", f.read(4))[0] # bool32 = int
            setup_info["bflipv"] = struct.unpack("<i", f.read(4))[0] # bool32
            setup_info["grid"] = struct.unpack("<I", f.read(4))[0]
            setup_info["frame_rate"] = struct.unpack("<I", f.read(4))[0]
            setup_info["shutter"] = struct.unpack("<I", f.read(4))[0]
            setup_info["edr_shutter"] = struct.unpack("<I", f.read(4))[0]
            setup_info["post_trigger"] = struct.unpack("<I", f.read(4))[0]
            setup_info["frame_delay"] = struct.unpack("<I", f.read(4))[0]
            setup_info["b_enable_color"] = struct.unpack("<i", f.read(4))[0] # bool32
            setup_info["camera_version"] = struct.unpack("<I", f.read(4))[0]
            setup_info["firmware_version"] = struct.unpack("<I", f.read(4))[0]
            setup_info["software_version"] = struct.unpack("<I", f.read(4))[0]
            setup_info["recording_time_zone"] = struct.unpack("<i", f.read(4))[0]
            setup_info["cfa"] = struct.unpack("<I", f.read(4))[0]
            setup_info["bright"] = struct.unpack("<i", f.read(4))[0]
            setup_info["contrast"] = struct.unpack("<i", f.read(4))[0]
            setup_info["gamma"] = struct.unpack("<i", f.read(4))[0]
            setup_info["res21"] = struct.unpack("<I", f.read(4))[0]
            setup_info["aut_exp_level"] = struct.unpack("<I", f.read(4))[0]
            setup_info["auto_exp_speed"] = struct.unpack("<I", f.read(4))[0]
            setup_info["auto_exp_rect"] = [struct.unpack("<i", f.read(4))[0],
                                           struct.unpack("<i", f.read(4))[0],
                                           struct.unpack("<i", f.read(4))[0],
                                           struct.unpack("<i", f.read(4))[0]] # left, top, right, bottom
            setup_info["wbgain"] = struct.unpack("f" * 2 * 4, f.read(4 * 2 * 4)) # white balance gain correction for red/blue. each wbgain has two floats
            setup_info["rotate"] = struct.unpack("<i", f.read(4))[0]
            setup_info["wbview"] = [struct.unpack("f", f.read(4))[0],
                                    struct.unpack("f", f.read(4))[0]]
            setup_info["realbpp"] = struct.unpack("<I", f.read(4))[0]
            setup_info["conv8min"] = struct.unpack("<I", f.read(4))[0]
            setup_info["conv8max"] = struct.unpack("<I", f.read(4))[0]
            setup_info["filter_code"] = struct.unpack("<i", f.read(4))[0]
            setup_info["filter_param"] = struct.unpack("<i", f.read(4))[0]
            setup_info["ufilter"] = [struct.unpack("<i", f.read(4))[0],
                                     struct.unpack("<i", f.read(4))[0],
                                     struct.unpack("<i", f.read(4))[0],
                                     struct.unpack("i" * 5 * 5, f.read(4 * 5 * 5))
                                     ] # imfilter = dim, shifts, bias, coefs
            setup_info["black_cal_sver"] = struct.unpack("<I", f.read(4))[0]
            setup_info["white_cal_sver"] = struct.unpack("<I", f.read(4))[0]
            setup_info["gray_cal_sver"] = struct.unpack("<I", f.read(4))[0]
            setup_info["b_stamp_time"] = struct.unpack("<i", f.read(4))[0] # bool32
            setup_info["sound_dest"] = struct.unpack("<I", f.read(4))[0]
            setup_info["frp_steps"] = struct.unpack("<I", f.read(4))[0]
            setup_info["frp_img_nr"] = struct.unpack("i" * 16, f.read(4 * 16))
            setup_info["frp_rate"] = struct.unpack("I" * 16, f.read(4 * 16))
            setup_info["frp_exp"] = struct.unpack("I" * 16, f.read(4 * 16))
            setup_info["mc_cnt"] = struct.unpack("<i", f.read(4))[0]
            setup_info["mc_percent"] = struct.unpack("f" * 64, f.read(4 * 64))
            setup_info["ci_calib"] = struct.unpack("<I", f.read(4))[0]
            setup_info["calib_width"] = struct.unpack("<I", f.read(4))[0]
            setup_info["calib_height"] = struct.unpack("<I", f.read(4))[0]
            setup_info["calib_rate"] = struct.unpack("<I", f.read(4))[0]
            setup_info["calib_exp"] = struct.unpack("<I", f.read(4))[0]
            setup_info["calib_edr"] = struct.unpack("<I", f.read(4))[0]
            setup_info["calib_temp"] = struct.unpack("<I", f.read(4))[0]
            setup_info["head_serial"] = struct.unpack("I" * 4, f.read(4 * 4))
            setup_info["range_code"] = struct.unpack("<I", f.read(4))[0]
            setup_info["range_size"] = struct.unpack("<I", f.read(4))[0]
            setup_info["decimation"] = struct.unpack("<I", f.read(4))[0]
            setup_info["master_serial"] = struct.unpack("<I", f.read(4))[0]
            setup_info["sensor"] = struct.unpack("<I", f.read(4))[0]
            setup_info["shutter_ns"] = struct.unpack("<I", f.read(4))[0]
            setup_info["edr_shutter_ns"] = struct.unpack("<I", f.read(4))[0]
            setup_info["frame_dealy_ns"] = struct.unpack("<I", f.read(4))[0]
            setup_info["im_pos_x_acq"] = struct.unpack("<I", f.read(4))[0]
            setup_info["im_pos_y_acq"] = struct.unpack("<I", f.read(4))[0]
            setup_info["im_width_acq"] = struct.unpack("<I", f.read(4))[0]
            setup_info["im_height_acq"] = struct.unpack("<I", f.read(4))[0]

            description_bytes = struct.unpack("c" * max_len_description, f.read(max_len_description))
            setup_info["description"] = "".join(c.decode() for c in description_bytes if c != b"\x00")

            setup_info["rising_edge"] = struct.unpack("<i", f.read(4))[0]  # bool32
            setup_info["filter_time"] = struct.unpack("<I", f.read(4))[0]
            setup_info["long_ready"] = struct.unpack("<i", f.read(4))[0]  # bool 32
            setup_info["shutter_off"] = struct.unpack("<i", f.read(4))[0]  #bool32
            setup_info["res4"] = struct.unpack("B" * 16, f.read(16))
            setup_info["b_meta_wb"] = struct.unpack("<i", f.read(4))[0]  #  bool 32
            setup_info["hue"] = struct.unpack("<i", f.read(4))[0]
            setup_info["black_level"] = struct.unpack("<i", f.read(4))[0]
            setup_info["white_level"] = struct.unpack("<i", f.read(4))[0]

            lens_description_bytes = struct.unpack("c" * 256, f.read(256))
            setup_info["lens_description"] = "".join(c.decode() for c in lens_description_bytes if c != b"\x00")
            setup_info["lens_aperture"] = struct.unpack("f", f.read(4))[0]
            setup_info["lens_focus_distance"] = struct.unpack("f", f.read(4))[0]
            setup_info["lens_focal_length"] = struct.unpack("f", f.read(4))[0]
            setup_info["f_offset"] = struct.unpack("f", f.read(4))[0]
            setup_info["f_gain"] = struct.unpack("f", f.read(4))[0]
            setup_info["f_saturation"] = struct.unpack("f", f.read(4))[0]
            setup_info["f_hue"] = struct.unpack("f", f.read(4))[0]
            setup_info["f_gamma"] = struct.unpack("f", f.read(4))[0]
            setup_info["f_gamma_r"] = struct.unpack("f", f.read(4))[0]
            setup_info["f_gamma_b"] = struct.unpack("f", f.read(4))[0]
            setup_info["f_flare"] = struct.unpack("f", f.read(4))[0]
            setup_info["f_pedestal_r"] = struct.unpack("f", f.read(4))[0]
            setup_info["f_pedestal_g"] = struct.unpack("f", f.read(4))[0]
            setup_info["f_pedestabl_b"] = struct.unpack("f", f.read(4))[0]
            setup_info["f_chroma"] = struct.unpack("f", f.read(4))[0]

            tone_label_bytes = struct.unpack("c" * 256, f.read(256))
            setup_info["tone_label"] = "".join(c.decode() for c in tone_label_bytes if c != b"\x00")

            setup_info["tone_points"] = struct.unpack("<i", f.read(4))[0]
            setup_info["f_tone"] = struct.unpack("f" * 32 * 2, f.read(4 * 32 * 2))

            user_matrix_label_bytes = struct.unpack("c" * 256, f.read(256))
            setup_info["user_matrix_label"] = "".join(c.decode() for c in user_matrix_label_bytes if c != b"\x00")

            setup_info["enable_matrices"] = struct.unpack("<i", f.read(4))[0]  # bool32
            setup_info["cm_user"] = struct.unpack("f" * 9, f.read(4 * 9))
            setup_info["enable_crop"] = struct.unpack("<i", f.read(4))[0]  # bool32
            setup_info["crop_rect"] = [struct.unpack("<i", f.read(4))[0],
                                       struct.unpack("<i", f.read(4))[0],
                                       struct.unpack("<i", f.read(4))[0],
                                       struct.unpack("<i", f.read(4))[0]] # left, top, right, bottom
            setup_info["enable_resample"] = struct.unpack("<i", f.read(4))[0]  # bool 32
            setup_info["resample_width"] = struct.unpack("<I", f.read(4))[0]
            setup_info["resample_height"] = struct.unpack("<I", f.read(4))[0]
            setup_info["f_gain16_8"] = struct.unpack("f", f.read(4))[0]
            setup_info["frp_shape"] = struct.unpack("I" * 16, f.read(4 * 16))[0]

            # todo ... how to unpack this?
            trig_tc = struct.unpack("c" * 8, f.read(4 + 4)) # uint8_t fields packed into 4 bytes
            setup_info["trig_tc"] = "".join(c.decode() for c in trig_tc)


            setup_info["fpb_rate"] = struct.unpack("f", f.read(4))[0]
            setup_info["ftc_rate"] = struct.unpack("f", f.read(4))[0]

            cine_name_bytes = struct.unpack("c" * 256, f.read(256))
            setup_info["cine_name"] = "".join(c.decode() for c in cine_name_bytes if c != b"\x00")

            setup_info["f_gain_r"] = struct.unpack("f", f.read(4))[0]
            setup_info["f_gain_g"] = struct.unpack("f", f.read(4))[0]
            setup_info["f_gain_b"] = struct.unpack("f", f.read(4))[0]
            setup_info["cm_calib"] = struct.unpack("f" * 9, f.read(4 * 9))
            setup_info["f_wb_temp"] = struct.unpack("f", f.read(4))[0]
            setup_info["f_wbc_c"] = struct.unpack("f", f.read(4))[0]

            calibration_info_bytes = struct.unpack("c" * 1024, f.read(1024))
            setup_info["calibration_info"] = "".join(c.decode() for c in calibration_info_bytes if c != b"\x00")

            optical_filter_bytes = struct.unpack("c" * 1024, f.read(1024))
            setup_info["optical_filter"] = "".join(c.decode() for c in optical_filter_bytes if c != b"\x00")

            gps_info_bytes = struct.unpack("c" * max_std_str_sz, f.read(max_std_str_sz))
            setup_info["gps_info"] = "".join(c.decode() for c in gps_info_bytes if c != b"\x00")

            uuid_bytes = struct.unpack("c" * max_std_str_sz, f.read(max_std_str_sz))
            setup_info["uuid"] = "".join(c.decode() for c in uuid_bytes if c != b"\x00")

            created_by_bytes = struct.unpack("c" * max_std_str_sz, f.read(max_std_str_sz))
            setup_info["created_by"] = "".join(c.decode() for c in created_by_bytes if c != b"\x00")

            setup_info["rec_bpp"] = struct.unpack("<I", f.read(4))[0]
            setup_info["lowest_format_bpp"] = struct.unpack("<H", f.read(2))[0]
            setup_info["lowest_format_q"] = struct.unpack("<H", f.read(2))[0]
            setup_info["ftoe"] = struct.unpack("f", f.read(4))[0]
            setup_info["log_mode"] = struct.unpack("<I", f.read(4))[0]

            camera_model_bytes = struct.unpack("c" * max_std_str_sz, f.read(max_std_str_sz))
            setup_info["camera_model"] = "".join(c.decode() for c in camera_model_bytes if c != b"\x00")

            setup_info["wb_type"] = struct.unpack("<I", f.read(4))[0]
            setup_info["f_decimation"] = struct.unpack("f", f.read(4))[0]
            setup_info["mag_serial"] = struct.unpack("<I", f.read(4))[0]
            setup_info["cs_serial"] = struct.unpack("<I", f.read(4))[0]
            setup_info["d_frame_rate"] = struct.unpack("d", f.read(8))[0]
            setup_info["sensor_mode"] = struct.unpack("<I", f.read(4))[0]


    def read_img(offset):

        with open(fname, "rb") as f:
            # read first image
            f.seek(int(offset))
            header_len = struct.unpack("<I", f.read(4))[0]
            img_start = int(offset) + header_len
            f.seek(img_start)
            bytes = np.frombuffer(f.read(bit_map_info["bi_size_image"]), dtype=np.uint8)

        # convert to 12L packed (two 12-bit pixels packed into three bytes)
        # a_uint8 = bytes[::3].astype(np.uint16)
        # b_uint8 = bytes[1::3].astype(np.uint16)
        # c_uint8 = bytes[2::3].astype(np.uint16)
        #
        # # middle byte contains least-significant bits of first integer and most-significant bits of second integer
        # first_int = (a_uint8 << 4) + (b_uint8 >> 4)
        # second_int = (np.bitwise_and(15, b_uint8) << 8) + c_uint8

        # img = np.stack((first_int, second_int), axis=1).reshape((bit_map_info["bi_height"], bit_map_info["bi_width"]))

        img = unpack12(bytes).reshape((bit_map_info["bi_height"], bit_map_info["bi_width"]))

        return np.expand_dims(img, axis=0)

    # combine all images into 3D dask array
    if end_index is None or end_index > len(offsets):
        end_index = len(offsets)

    offsets_da = da.from_array(offsets[start_index:end_index], chunks=(1,))

    imgs = da.map_blocks(read_img,
                         offsets_da,
                         new_axis=(1, 2),
                         chunks=(1, bit_map_info["bi_height"], bit_map_info["bi_width"]),
                         dtype=np.uint16,
                         )

    # imgs = da.concatenate([da.from_delayed(dask.delayed(read_img)(o),
    #                                  shape=(bit_map_info["bi_height"], bit_map_info["bi_width"]),
    #                                  dtype="uint16") for o in offsets[start_index:end_index]], axis=0)

    metadata = {"header_data": header,
                "bitmap_info": bit_map_info,
                "setup_info": setup_info}

    return imgs, metadata

def pack12(data):
    """
    Convert data from uint16 integers to 12-bit packed format

    :param data: even sized array of uint16
    :return bytes:
    """

    # most significant 8 bits of first integer
    first_bit = (data[::2] >> 4).astype(np.uint8)
    # least significant 4 bits of first integer and most significant 4 bits of second
    second_bit = ((np.bitwise_and(15, data[::2]) << 4) + (data[1::2] >> 8)).astype(np.uint8)
    # least significant 8 bits of second integer
    third_bit = np.bitwise_and(255, data[1::2]).astype(np.uint8)

    return np.stack((first_bit, second_bit, third_bit), axis=1).ravel()

def unpack12(data: np.ndarray) -> np.ndarray:
    """
    Convert data from 12-bit packed integers to uint16 integers

    :param data: an array of uint8 integers
    :return img: an array of uint16 integers
    """

    # convert from 12L packed (two 12-bit pixels packed into three bytes)
    a_uint8 = data[::3].astype(np.uint16)
    b_uint8 = data[1::3].astype(np.uint16)
    c_uint8 = data[2::3].astype(np.uint16)

    # middle byte contains least-significant bits of first integer
    # and most-significant bits of second integer
    first_int = (a_uint8 << 4) + (b_uint8 >> 4)
    second_int = (np.bitwise_and(15, b_uint8) << 8) + c_uint8

    img = np.stack((first_int, second_int), axis=1).ravel()

    return img
