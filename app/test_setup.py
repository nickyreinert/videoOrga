"""
Test Setup Script
Verifies that all dependencies are installed correctly
"""

import sys


def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    tests = [
        ("OpenCV", "cv2"),
        ("PIL/Pillow", "PIL"),
        ("NumPy", "numpy"),
        ("PyTorch", "torch"),
        ("Transformers", "transformers"),
    ]
    
    failed = []
    
    for name, module in tests:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - {e}")
            failed.append(name)
    
    return len(failed) == 0, failed


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"  ✓ CUDA is available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("  ✗ CUDA is not available")
            print("  Note: CPU mode will work but will be slower")
            return False
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False


def test_file_operations():
    """Test file and database operations"""
    print("\nTesting file operations...")
    
    try:
        from db_handler import DatabaseHandler, parse_datetime_from_filename
        
        # Test datetime parsing
        test_filenames = {
            "VID_20231215_142530.mp4": "2023-12-15",
            "2023-12-15.mp4": "2023-12-15",
            "random_video.mp4": None
        }
        
        parsing_ok = True
        for filename, expected in test_filenames.items():
            result = parse_datetime_from_filename(filename)
            matches = (result is None and expected is None) or (result and result.startswith(expected))
            if matches:
                print(f"  ✓ Parsed '{filename}'")
            else:
                print(f"  ✗ Failed to parse '{filename}': got {result}, expected {expected}")
                parsing_ok = False
        
        # Test database creation
        import os
        test_db = "test_setup.db"
        
        try:
            db = DatabaseHandler(test_db)
            print(f"  ✓ Database creation successful")
            
            # Get stats
            stats = db.get_statistics()
            print(f"  ✓ Database queries working")
            
            db.close()
            
            # Clean up
            if os.path.exists(test_db):
                os.remove(test_db)
                print(f"  ✓ Database cleanup successful")
            
            return parsing_ok
            
        except Exception as e:
            print(f"  ✗ Database test failed: {e}")
            return False
    
    except Exception as e:
        print(f"  ✗ File operations test failed: {e}")
        return False


def test_video_processing():
    """Test basic video processing capabilities"""
    print("\nTesting video processing...")
    
    try:
        from frame_extractor import FrameExtractor
        
        extractor = FrameExtractor(num_frames=4)
        print(f"  ✓ FrameExtractor initialized")
        
        # Note: We can't test actual video processing without a video file
        print(f"  ℹ Actual video processing requires a video file")
        print(f"  Run: python video_tagger.py <your_video.mp4>")
        
        return True
    
    except Exception as e:
        print(f"  ✗ Video processing test failed: {e}")
        return False


def test_ai_models():
    """Test AI model loading (optional, can be slow)"""
    print("\nTesting AI models (optional)...")
    print("  ℹ Skipping model download test (run manually to test)")
    print("  To test: python -c 'from ai_analyzer import AIAnalyzer; a = AIAnalyzer(\"blip\"); a.load_model()'")
    return True


def main():
    """Run all tests"""
    print("="*60)
    print("Video Auto-Tagger - Setup Verification")
    print("="*60)
    print()
    
    results = {}
    
    # Run tests
    results['imports'], failed_imports = test_imports()
    results['cuda'] = test_cuda()
    results['files'] = test_file_operations()
    results['video'] = test_video_processing()
    results['ai'] = test_ai_models()
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name.upper()}")
    
    # Overall result
    critical_tests = ['imports', 'files', 'video']
    all_critical_passed = all(results[t] for t in critical_tests)
    
    print()
    if all_critical_passed:
        print("✓ Setup is ready!")
        if not results['cuda']:
            print("⚠ Warning: CUDA not available, will use CPU (slower)")
        print("\nNext steps:")
        print("  1. python video_tagger.py your_video.mp4")
        print("  2. python video_tagger.py /path/to/videos --recursive")
    else:
        print("✗ Setup has issues. Please fix the failed tests above.")
        if not results['imports']:
            print("\nMissing packages. Install with:")
            print("  pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()