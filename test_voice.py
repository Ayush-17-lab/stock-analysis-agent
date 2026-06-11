#!/usr/bin/env python3
"""
Test script for voice input functionality
Run this to test if voice input is working properly
"""

def test_voice_dependencies():
    """Test if all voice dependencies are installed"""
    print("🔍 Testing voice input dependencies...")
    
    try:
        import speech_recognition as sr
        print("✅ SpeechRecognition installed")
    except ImportError:
        print("❌ SpeechRecognition not installed")
        return False
    
    try:
        import pyaudio
        print("✅ PyAudio installed")
    except ImportError:
        print("❌ PyAudio not installed")
        return False
    
    try:
        import pocketsphinx
        print("✅ Pocketsphinx installed")
    except ImportError:
        print("⚠️ Pocketsphinx not installed (optional)")
    
    return True

def test_microphone():
    """Test if microphone is accessible"""
    print("\n🎤 Testing microphone access...")
    
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        
        with sr.Microphone() as source:
            print("✅ Microphone accessible")
            return True
    except Exception as e:
        print(f"❌ Microphone error: {e}")
        return False

def test_voice_recognition():
    """Test voice recognition"""
    print("\n🎯 Testing voice recognition...")
    
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        
        print("Speak something in 3 seconds...")
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
            
            try:
                text = recognizer.recognize_google(audio)
                print(f"✅ Recognized: {text}")
                return True
            except sr.UnknownValueError:
                print("⚠️ Could not understand audio")
                return False
            except sr.RequestError as e:
                print(f"❌ Recognition error: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Voice recognition error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Voice Input Test Suite")
    print("=" * 40)
    
    # Test dependencies
    deps_ok = test_voice_dependencies()
    
    if deps_ok:
        # Test microphone
        mic_ok = test_microphone()
        
        if mic_ok:
            # Test recognition
            rec_ok = test_voice_recognition()
            
            if rec_ok:
                print("\n🎉 All tests passed! Voice input is ready to use.")
            else:
                print("\n⚠️ Voice recognition failed. Check your internet connection.")
        else:
            print("\n❌ Microphone not accessible. Check permissions.")
    else:
        print("\n❌ Dependencies missing. Install with:")
        print("pip install SpeechRecognition PyAudio pocketsphinx")
