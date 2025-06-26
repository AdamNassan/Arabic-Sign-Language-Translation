from featuresExtractor import features

# Video settings
diVideoSet = {
    "dataset": "ArabSign",
    "nClasses": 50,
    "nFramesNorm": 80,
    "nMinDim": 224,
    "tuShape": (224, 224),
    "nFpsAvg": 30,
    "nFramesAvg": 50,
    "fDurationAvg": 2.0,
    "reshape_input": False
}

# Feature extractor settings
diFeature = {
    "model": "vgg16",
    "tuInputShape": (224, 224, 3),
    "tuOutputShape": (4096,)
}

# Paths
frames_path = "./frames/Color"  # Path to extracted frames
features_path = "./features/images/vgg16/color"  # Path for saving features

# Extract features
print('============== START OF FEATURES EXTRACTION ====================')
extractor = features(frames_path, features_path, diFeature)
extractor.extractFeatures()
print('============== END OF FEATURES EXTRACTION ====================') 