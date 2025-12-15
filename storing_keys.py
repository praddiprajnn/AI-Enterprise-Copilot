#This file is used when we use gemini api keys but in this project we are not using any api keys so this file is redundant
GEMINI_API_KEY = ""
laptopGEMINI_API_KEY=""
def validate_keys():
    """Validate that required API keys are present"""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("⚠️  WARNING: Please set your GEMINI_API_KEY in storing_keys.py")
        print("Get it from: https://makersuite.google.com/app/apikey")
        return False
    return True

if __name__ == "__main__":
    validate_keys()
    