#!/usr/bin/env python3
"""
Minimal test script to verify KGGLM functionality after distillation.
This script tests that KGGLM can be imported and basic functionality works.
"""

import sys
import traceback

def test_imports():
    """Test that all required KGGLM components can be imported."""
    print("Testing imports...")
    
    try:
        # Test core model imports
        from hopwise.model.path_language_modeling_recommender.kgglm import KGGLM
        from hopwise.model.path_language_modeling_recommender.pearlm import PEARLM
        print("✓ KGGLM and PEARLM models imported successfully")
        
        # Test abstract recommender classes
        from hopwise.model.abstract_recommender import (
            AbstractRecommender,
            KnowledgeRecommender, 
            PathLanguageModelingRecommender,
            ExplainablePathLanguageModelingRecommender
        )
        print("✓ Abstract recommender classes imported successfully")
        
        # Test dataset classes
        from hopwise.data.dataset import Dataset, KnowledgeBasedDataset, KnowledgePathDataset
        print("✓ Dataset classes imported successfully")
        
        # Test trainer classes
        from hopwise.trainer import (
            AbstractTrainer,
            Trainer,
            ExplainableTrainer,
            PretrainTrainer,
            HFPathLanguageModelingTrainer,
            KGGLMTrainer
        )
        print("✓ Trainer classes imported successfully")
        
        # Test utilities
        from hopwise.utils import PathLanguageModelingTokenType, GenerationOutputs
        print("✓ Utility classes imported successfully")
        
        # Test quick start
        from hopwise.quick_start import run
        print("✓ Quick start function imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_model_class_structure():
    """Test that KGGLM class has the expected structure."""
    print("\nTesting model class structure...")

    try:
        # Import KGGLM class
        from hopwise.model.path_language_modeling_recommender.kgglm import KGGLM
        from hopwise.model.path_language_modeling_recommender.pearlm import PEARLM

        # Check that KGGLM inherits from PEARLM
        assert issubclass(KGGLM, PEARLM)
        print("✓ KGGLM correctly inherits from PEARLM")

        # Check that KGGLM has expected class attributes
        assert hasattr(KGGLM, '__init__')
        print("✓ KGGLM has __init__ method")

        # Check that PEARLM has expected parent classes
        from hopwise.model.abstract_recommender import ExplainablePathLanguageModelingRecommender
        assert issubclass(PEARLM, ExplainablePathLanguageModelingRecommender)
        print("✓ PEARLM correctly inherits from ExplainablePathLanguageModelingRecommender")

        return True

    except Exception as e:
        print(f"✗ Model class structure test failed: {e}")
        traceback.print_exc()
        return False

def test_cli_import():
    """Test that CLI can be imported."""
    print("\nTesting CLI import...")
    
    try:
        from hopwise.cli import cli
        print("✓ CLI imported successfully")
        return True
        
    except ImportError as e:
        print(f"✗ CLI import failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("KGGLM Distillation Validation Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_class_structure,
        test_cli_import,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! KGGLM distillation successful.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
