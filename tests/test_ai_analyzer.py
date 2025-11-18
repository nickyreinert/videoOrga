import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add app directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ai_analyzer import AIAnalyzer

class TestAIAnalyzerStopwords:
    
    @pytest.fixture
    def analyzer_en(self):
        """Fixture for English analyzer"""
        with patch('app.ai_analyzer.AIAnalyzer.load_model'):
            analyzer = AIAnalyzer(tag_language='en', device='cpu')
            return analyzer

    @pytest.fixture
    def analyzer_de(self):
        """Fixture for German analyzer"""
        with patch('app.ai_analyzer.AIAnalyzer.load_model'):
            analyzer = AIAnalyzer(tag_language='de', device='cpu')
            return analyzer

    def test_stopword_loading_en(self, analyzer_en):
        """Test that English stopwords are loaded"""
        assert 'the' in analyzer_en.stopwords
        assert 'is' in analyzer_en.stopwords  # Common verb
        assert 'and' in analyzer_en.stopwords

    def test_stopword_loading_de(self, analyzer_de):
        """Test that German stopwords are loaded"""
        assert 'der' in analyzer_de.stopwords
        assert 'ist' in analyzer_de.stopwords  # Common verb
        assert 'und' in analyzer_de.stopwords
        assert 'gibt' in analyzer_de.stopwords # Added common verb

    def test_extract_tags_basic_en(self, analyzer_en):
        """Test basic tag extraction in English"""
        text = "The quick brown fox jumps over the lazy dog."
        tags = analyzer_en._extract_tags_from_text(text)
        
        # 'the', 'over' should be removed
        assert 'quick' in tags
        assert 'brown' in tags
        assert 'fox' in tags
        assert 'jumps' in tags
        assert 'lazy' in tags
        assert 'the' not in tags
        assert 'over' not in tags # usually a stopword

    def test_extract_tags_verbs_en(self, analyzer_en):
        """Test removal of common English verbs"""
        text = "This video is showing a man who was walking and has a dog."
        tags = analyzer_en._extract_tags_from_text(text)
        
        assert 'video' in tags
        assert 'showing' in tags # might be kept or removed depending on list, usually kept as it's content
        assert 'man' in tags
        assert 'walking' in tags
        assert 'dog' in tags
        
        # Verbs to be removed
        assert 'is' not in tags
        assert 'was' not in tags
        assert 'has' not in tags

    def test_extract_tags_verbs_de(self, analyzer_de):
        """Test removal of common German verbs"""
        text = "Das Video zeigt einen Mann der geht und einen Hund hat. Es gibt auch eine Katze."
        tags = analyzer_de._extract_tags_from_text(text)
        
        assert 'video' in tags
        assert 'zeigt' in tags # meaningful verb
        assert 'mann' in tags
        assert 'hund' in tags
        assert 'katze' in tags
        
        # Stopwords/Verbs to be removed
        assert 'das' not in tags
        assert 'der' not in tags
        assert 'und' not in tags
        assert 'einen' not in tags
        assert 'hat' not in tags
        assert 'gibt' not in tags
        assert 'eine' not in tags
        assert 'auch' not in tags # usually stopword

    def test_short_tag_filtering(self, analyzer_en):
        """Test filtering of short tags"""
        text = "a ab abc abcd"
        tags = analyzer_en._extract_tags_from_text(text)
        
        assert 'a' not in tags
        assert 'ab' not in tags
        assert 'abc' not in tags # length > 2 required, so 3 is excluded? check implementation: len(tag) > 2
        # Implementation says: if tag and len(tag) > 2. So 'abc' (len 3) is kept.
        assert 'abc' in tags 
        assert 'abcd' in tags

    def test_nonsense_filtering(self, analyzer_en):
        """Test filtering of nonsense tags (repeated chars)"""
        text = "good tag nnn mmm ooo"
        tags = analyzer_en._extract_tags_from_text(text)
        
        assert 'good' in tags
        assert 'tag' in tags
        assert 'nnn' not in tags
        assert 'mmm' not in tags
        assert 'ooo' not in tags

    def test_custom_stopwords(self):
        """Test adding custom stopwords"""
        with patch('app.ai_analyzer.AIAnalyzer.load_model'):
            custom = ['specific', 'word']
            analyzer = AIAnalyzer(tag_language='en', stopwords=custom, device='cpu')
            
            text = "This is a specific word test."
            tags = analyzer._extract_tags_from_text(text)
            
            assert 'test' in tags
            assert 'specific' not in tags
            assert 'word' not in tags
