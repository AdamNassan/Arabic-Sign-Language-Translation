from featuresExtractor import features

##
diVideoSet = {"dataset" : "ArabSign",
    "nClasses" : 50,   # number of classes
    "nFramesNorm" : 80,    # number of frames per video
    "nMinDim" : 299,   # smaller dimension of saved video-frames
    "tuShape" : (299, 299), # height, width
    "nFpsAvg" : 30,
    "nFramesAvg" : 50, 
    "fDurationAvg" : 2.0,# seconds 
    "reshape_input": False}  #True: if the raw input is different from the requested shape for the model

# feature extractor 
diFeature = {"model" : "inception",
    "tuInputShape" : (299, 299, 3),
    "tuOutputShape" : (2048, )}  # Inception output features size

# Video and Frames paths
extractFrames = False
FramesPath  = 'frames/color'

# Features destination path
extractFeatures = True
destFeaturesPath = 'features/color'

# Extract features from video frames
if extractFeatures:
    extractor = features(FramesPath, destFeaturesPath, diFeature)
    print('============== START OF FEATURES EXTRACTION ====================')
    extractor.extractFeatures()
    print('============== END OF FEATURES EXTRACTION ====================')

# Model train and test



