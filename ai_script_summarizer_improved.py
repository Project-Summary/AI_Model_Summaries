# ai_script_summarizer_improved.py
import json
import pickle
import re
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import nltk
from collections import defaultdict, Counter
import pymongo
from pymongo import MongoClient
from bson import ObjectId
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

class ImprovedScriptSummarizerAI:
    def __init__(self, mongodb_uri="mongodb+srv://nguyenvanninh:MM1onTUoA1sF4AnZ@onebizai.tt3rs.mongodb.net/", db_name="test"):
        """
        AI tóm tắt kịch bản cải thiện với MongoDB
        """
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        
        # Collections theo schema mới
        self.scripts_collection = self.db.scripts  # Collection Script
        self.feedbacks_collection = self.db.feedbacks  # Collection Feedback  
        self.categories_collection = self.db.categories  # Collection Category
        
        # Collections cho AI training
        self.training_metrics_collection = self.db.training_metrics
        self.learned_patterns_collection = self.db.learned_patterns
        self.ai_summaries_collection = self.db.ai_summaries  # Lưu kết quả AI tóm tắt
        
        # AI Models
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words=self._get_vietnamese_stopwords(),
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
        
        self.quality_predictor = GradientBoostingRegressor(
            n_estimators=150, 
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.genre_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        
        # Episode templates với độ dài A4 (khoảng 500-600 từ)
        self.episode_templates = {
            'drama': {
                'intro': {'ratio': 0.15, 'target_words': 150},
                'development': {'ratio': 0.35, 'target_words': 200},
                'climax': {'ratio': 0.30, 'target_words': 180},
                'resolution': {'ratio': 0.20, 'target_words': 120}
            },
            'comedy': {
                'setup': {'ratio': 0.25, 'target_words': 150},
                'conflict': {'ratio': 0.50, 'target_words': 250},
                'resolution': {'ratio': 0.25, 'target_words': 150}
            },
            'action': {
                'setup': {'ratio': 0.20, 'target_words': 120},
                'action_sequence': {'ratio': 0.60, 'target_words': 300},
                'aftermath': {'ratio': 0.20, 'target_words': 130}
            },
            'romance': {
                'meeting': {'ratio': 0.30, 'target_words': 180},
                'development': {'ratio': 0.40, 'target_words': 220},
                'resolution': {'ratio': 0.30, 'target_words': 150}
            }
        }
        
        self.knowledge_base = []
        self._load_models()
        self._create_indexes()
        
    def _create_indexes(self):
        """Tạo indexes cho MongoDB để tối ưu performance"""
        try:
            # Indexes cho AI collections
            self.ai_summaries_collection.create_index([("created_at", -1)])
            self.ai_summaries_collection.create_index([("script_id", 1)])
            self.ai_summaries_collection.create_index([("genre", 1)])
            self.ai_summaries_collection.create_index([("quality_score", -1)])
            
            self.training_metrics_collection.create_index([("trained_at", -1)])
            self.learned_patterns_collection.create_index([("learned_at", -1)])
            
            print("✅ MongoDB indexes created successfully")
        except Exception as e:
            print(f"⚠️ Error creating indexes: {e}")
    
    def _get_vietnamese_stopwords(self):
        """Tạo danh sách stopwords tiếng Việt mở rộng"""
        vietnamese_stopwords = [
            'là', 'của', 'và', 'có', 'cho', 'với', 'trong', 'không', 'được', 'một',
            'này', 'đó', 'những', 'các', 'để', 'từ', 'trên', 'về', 'khi', 'nếu',
            'mà', 'như', 'đã', 'sẽ', 'bị', 'hay', 'thì', 'hoặc', 'nhưng', 'vì',
            'tại', 'theo', 'giữa', 'sau', 'trước', 'ngoài', 'dưới', 'lên', 'xuống',
            'cũng', 'đều', 'phải', 'chỉ', 'lại', 'còn', 'hơn', 'nữa', 'khác',
            'nhiều', 'ít', 'lớn', 'nhỏ', 'cao', 'thấp', 'xa', 'gần', 'mới', 'cũ'
        ]
        
        try:
            english_stopwords = list(stopwords.words('english'))
            return vietnamese_stopwords + english_stopwords
        except:
            return vietnamese_stopwords
    
    def get_categories_from_db(self) -> List[Dict]:
        """Lấy danh sách categories từ DB"""
        try:
            categories = list(self.categories_collection.find(
                {"isActive": True},
                {"name": 1, "slug": 1, "description": 1, "type": 1}
            ))
            return categories
        except Exception as e:
            print(f"❌ Error fetching categories: {e}")
            return []
    
    def detect_genre_from_categories(self, text: str) -> Tuple[str, float, Optional[ObjectId]]:
        """Phát hiện thể loại dựa trên categories trong DB"""
        text_lower = text.lower()
        categories = self.get_categories_from_db()
        
        if not categories:
            # Fallback to default detection
            genre, confidence = self.detect_genre(text)
            return genre, confidence, None
        
        category_scores = {}
        
        for category in categories:
            category_name = category['name'].lower()
            category_slug = category['slug'].lower()
            category_desc = category.get('description', '').lower()
            
            score = 0
            
            # Tìm kiếm trong text
            if category_name in text_lower:
                score += 3
            if category_slug in text_lower:
                score += 2
            
            # Tìm kiếm keywords liên quan
            if 'hành động' in category_name or 'action' in category_name:
                action_keywords = ['đánh nhau', 'chiến đấu', 'súng', 'nổ', 'rượt đuổi', 'hành động']
                score += sum(2 for keyword in action_keywords if keyword in text_lower)
            
            elif 'tình cảm' in category_name or 'romance' in category_name:
                romance_keywords = ['yêu', 'tình yêu', 'hẹn hò', 'cưới', 'lãng mạn']
                score += sum(2 for keyword in romance_keywords if keyword in text_lower)
            
            elif 'hài' in category_name or 'comedy' in category_name:
                comedy_keywords = ['hài hước', 'vui nhộn', 'cười', 'hài', 'buồn cười']
                score += sum(2 for keyword in comedy_keywords if keyword in text_lower)
            
            elif 'kinh dị' in category_name or 'thriller' in category_name:
                thriller_keywords = ['bí ẩn', 'giết', 'chết', 'nguy hiểm', 'sợ hãi', 'kinh dị']
                score += sum(2 for keyword in thriller_keywords if keyword in text_lower)
            
            if score > 0:
                category_scores[category['_id']] = {
                    'score': score,
                    'name': category['name'],
                    'slug': category['slug']
                }
        
        if not category_scores:
            # Fallback to default detection
            genre, confidence = self.detect_genre(text)
            return genre, confidence, None
        
        # Tìm category có điểm cao nhất
        best_category_id = max(category_scores, key=lambda x: category_scores[x]['score'])
        best_category = category_scores[best_category_id]
        
        total_score = sum(cat['score'] for cat in category_scores.values())
        confidence = best_category['score'] / total_score if total_score > 0 else 0.5
        
        return best_category['name'], min(confidence, 1.0), best_category_id
    
    def detect_genre(self, text: str) -> Tuple[str, float]:
        """Phát hiện thể loại với confidence score (fallback method)"""
        text_lower = text.lower()
        
        genre_keywords = {
            'action': {
                'keywords': ['đánh nhau', 'chiến đấu', 'súng', 'nổ', 'rượt đuổi', 'hành động', 'phiêu lưu', 'tấn công', 'bạo lực'],
                'weight': [3, 3, 2, 2, 2, 1, 1, 2, 2]
            },
            'drama': {
                'keywords': ['cảm xúc', 'tình cảm', 'gia đình', 'nước mắt', 'đau khổ', 'bi kịch', 'xúc động', 'tâm lý'],
                'weight': [2, 2, 3, 2, 2, 3, 2, 2]
            },
            'comedy': {
                'keywords': ['hài hước', 'vui nhộn', 'cười', 'hài', 'buồn cười', 'vui vẻ', 'hóm hỉnh', 'dí dỏm'],
                'weight': [3, 2, 3, 3, 2, 1, 2, 2]
            },
            'romance': {
                'keywords': ['yêu', 'tình yêu', 'hẹn hò', 'cưới', 'lãng mạn', 'tình cảm', 'đám cưới', 'valentine'],
                'weight': [3, 3, 2, 2, 3, 2, 2, 1]
            },
            'thriller': {
                'keywords': ['bí ẩn', 'giết', 'chết', 'nguy hiểm', 'sợ hãi', 'kinh dị', 'căng thẳng', 'hồi hộp'],
                'weight': [2, 3, 2, 2, 2, 3, 2, 2]
            },
            'fantasy': {
                'keywords': ['phép thuật', 'ma thuật', 'thần tiên', 'rồng', 'phù thủy', 'siêu nhiên', 'kỳ ảo'],
                'weight': [3, 3, 2, 2, 2, 2, 2]
            }
        }
        
        genre_scores = {}
        for genre, data in genre_keywords.items():
            score = 0
            for keyword, weight in zip(data['keywords'], data['weight']):
                if keyword in text_lower:
                    score += weight
            genre_scores[genre] = score
        
        if not genre_scores or max(genre_scores.values()) == 0:
            return 'drama', 0.5
        
        best_genre = max(genre_scores, key=genre_scores.get)
        total_score = sum(genre_scores.values())
        confidence = genre_scores[best_genre] / total_score if total_score > 0 else 0.5
        
        return best_genre, min(confidence, 1.0)
    
    def extract_key_scenes(self, text: str) -> List[Dict]:
        """Trích xuất cảnh quan trọng với thuật toán cải thiện"""
        sentences = sent_tokenize(text)
        
        if len(sentences) == 0:
            return []
        
        # Tính TF-IDF scores
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            sentence_scores = np.mean(tfidf_matrix.toarray(), axis=1)
        except:
            sentence_scores = np.ones(len(sentences))
        
        # Keywords quan trọng với trọng số
        important_keywords = {
            'plot': ['bất ngờ', 'shock', 'đột nhiên', 'bỗng nhiên', 'tình tiết'],
            'structure': ['bắt đầu', 'mở đầu', 'cuối cùng', 'kết thúc', 'chấm dứt'],
            'conflict': ['xung đột', 'tranh cãi', 'đối đầu', 'chống lại', 'cãi nhau'],
            'emotion': ['cảm xúc', 'xúc động', 'tức giận', 'vui mừng', 'buồn bã']
        }
        
        scenes = []
        for i, sentence in enumerate(sentences):
            importance = sentence_scores[i]
            
            # Tăng điểm dựa trên keywords
            for category, keywords in important_keywords.items():
                for keyword in keywords:
                    if keyword in sentence.lower():
                        if category == 'plot':
                            importance += 0.8
                        elif category == 'structure':
                            importance += 0.6
                        elif category == 'conflict':
                            importance += 0.7
                        elif category == 'emotion':
                            importance += 0.5
            
            # Tăng điểm cho câu có độ dài vừa phải
            sentence_length = len(sentence.split())
            if 10 <= sentence_length <= 30:
                importance += 0.3
            
            scenes.append({
                'text': sentence,
                'position': i / len(sentences),
                'importance': importance,
                'scene_type': self._classify_scene_type(sentence),
                'word_count': sentence_length
            })
        
        # Sắp xếp theo độ quan trọng
        scenes.sort(key=lambda x: x['importance'], reverse=True)
        return scenes
    
    def _classify_scene_type(self, text: str) -> str:
        """Phân loại loại cảnh chi tiết hơn"""
        text_lower = text.lower()
        
        scene_patterns = {
            'dialogue': ['nói', 'hỏi', 'trả lời', 'nói chuyện', 'thì thầm', 'hét lên', 'thốt lên'],
            'action': ['đánh', 'chạy', 'nhảy', 'đuổi', 'tấn công', 'bỏ chạy', 'lao tới'],
            'internal': ['nghĩ', 'cảm thấy', 'tâm trạng', 'suy nghĩ', 'nhớ lại', 'hối hận'],
            'description': ['cảnh', 'khung cảnh', 'bối cảnh', 'môi trường', 'không gian'],
            'narrative': []  # Default
        }
        
        for scene_type, keywords in scene_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return scene_type
        
        return 'narrative'
    
    def create_a4_episode_summary(self, scenes: List[Dict], episode_num: int, genre: str) -> Dict:
        """Tạo tóm tắt tập phim độ dài 1 trang A4 (550-650 từ)"""
        if not scenes:
            return {
                'episode_number': episode_num,
                'title': f"Tập {episode_num}",
                'summary': f"Tập {episode_num}: Nội dung đang được cập nhật.",
                'word_count': 0,
                'sections': {}
            }
        
        template = self.episode_templates.get(genre, self.episode_templates['drama'])
        
        # Phân chia cảnh theo vị trí
        intro_scenes = [s for s in scenes if s['position'] < 0.3]
        middle_scenes = [s for s in scenes if 0.3 <= s['position'] < 0.7]
        end_scenes = [s for s in scenes if s['position'] >= 0.7]
        
        sections = {}
        total_words = 0
        
        # Tạo từng phần theo template
        for section_name, section_config in template.items():
            target_words = section_config['target_words']
            
            if section_name in ['intro', 'setup', 'meeting']:
                section_scenes = intro_scenes[:3]
            elif section_name in ['development', 'conflict', 'action_sequence']:
                section_scenes = middle_scenes[:4]
            else:  # resolution, aftermath
                section_scenes = end_scenes[:2]
            
            section_summary = self._create_detailed_section_summary(
                section_scenes, section_name, target_words
            )
            
            sections[section_name] = {
                'title': section_name.title(),
                'content': section_summary,
                'word_count': len(section_summary.split())
            }
            
            total_words += len(section_summary.split())
        
        # Tạo tóm tắt hoàn chỉnh
        full_summary = self._combine_sections_to_a4(sections, episode_num)
        
        return {
            'episode_number': episode_num,
            'title': f"Tập {episode_num}",
            'summary': full_summary,
            'word_count': len(full_summary.split()),
            'sections': sections,
            'genre': genre,
            'target_length': '1 trang A4 (550-650 từ)'
        }
    
    def _create_detailed_section_summary(self, scenes: List[Dict], section_name: str, target_words: int) -> str:
        """Tạo tóm tắt chi tiết cho một phần"""
        if not scenes:
            return f"Phần {section_name}: Nội dung đang được phát triển."
        
        # Trích xuất thông tin chính
        main_events = []
        characters = set()
        locations = set()
        emotions = []
        
        for scene in scenes:
            text = scene['text']
            
            # Trích xuất nhân vật (từ viết hoa)
            potential_chars = re.findall(r'\b[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+', text)
            characters.update([char for char in potential_chars if len(char) > 2 and char not in ['Tôi', 'Anh', 'Chị', 'Em']])
            
            # Trích xuất địa điểm
            location_keywords = ['tại', 'ở', 'trong', 'ngoài', 'trên', 'dưới']
            for keyword in location_keywords:
                if keyword in text.lower():
                    words_after = text.lower().split(keyword)[1:2]
                    if words_after:
                        potential_location = words_after[0].split()[:3]
                        locations.add(' '.join(potential_location).strip(' .,!?'))
            
            # Trích xuất cảm xúc
            emotion_keywords = ['vui', 'buồn', 'tức giận', 'hạnh phúc', 'lo lắng', 'sợ hãi', 'bất ngờ']
            for emotion in emotion_keywords:
                if emotion in text.lower():
                    emotions.append(emotion)
            
            # Trích xuất sự kiện chính (câu ngắn gọn)
            if len(text.split()) <= 25:  # Câu ngắn, có thể là event chính
                main_events.append(text[:100] + '...' if len(text) > 100 else text)
        
        # Tạo tóm tắt dựa trên thông tin đã trích xuất
        summary_parts = []
        
        # Nhân vật
        if characters:
            char_list = list(characters)[:3]  # Lấy tối đa 3 nhân vật
            summary_parts.append(f"Nhân vật chính: {', '.join(char_list)}")
        
        # Sự kiện
        if main_events:
            event_summary = '. '.join(main_events[:2])  # Lấy 2 sự kiện chính
            summary_parts.append(f"Diễn biến: {event_summary}")
        
        # Cảm xúc
        if emotions:
            unique_emotions = list(set(emotions))[:2]
            summary_parts.append(f"Tâm trạng: {', '.join(unique_emotions)}")
        
        # Địa điểm
        if locations:
            location_list = [loc for loc in locations if len(loc.strip()) > 3][:2]
            if location_list:
                summary_parts.append(f"Bối cảnh: {', '.join(location_list)}")
        
        # Kết hợp và điều chỉnh độ dài
        base_summary = '. '.join(summary_parts) + '.'
        
        # Mở rộng nếu cần để đạt target_words
        current_words = len(base_summary.split())
        if current_words < target_words * 0.8:  # Nếu quá ngắn
            # Thêm chi tiết từ scenes
            additional_details = []
            for scene in scenes[:2]:
                if scene['scene_type'] == 'dialogue':
                    additional_details.append(f"Đối thoại: {scene['text'][:80]}...")
                elif scene['scene_type'] == 'action':
                    additional_details.append(f"Hành động: {scene['text'][:80]}...")
            
            if additional_details:
                base_summary += ' ' + ' '.join(additional_details)
        
        return base_summary
    
    def _combine_sections_to_a4(self, sections: Dict, episode_num: int) -> str:
        """Kết hợp các phần thành tóm tắt A4 hoàn chỉnh"""
        header = f"=== TẬP {episode_num} ===\n\n"
        
        content_parts = []
        for section_name, section_data in sections.items():
            section_title = section_data['title']
            section_content = section_data['content']
            
            formatted_section = f"【{section_title}】\n{section_content}\n"
            content_parts.append(formatted_section)
        
        # Thêm thống kê cuối
        total_words = sum(section['word_count'] for section in sections.values())
        footer = f"\n--- Tổng số từ: {total_words} ---"
        
        full_content = header + '\n'.join(content_parts) + footer
        
        # Đảm bảo độ dài phù hợp (550-650 từ)
        current_words = len(full_content.split())
        if current_words > 650:
            # Cắt bớt nếu quá dài
            words = full_content.split()[:650]
            full_content = ' '.join(words) + '...'
        elif current_words < 550:
            # Thêm nội dung nếu quá ngắn
            padding = "\n\nGhi chú: Tập này chứa nhiều tình tiết quan trọng và cảm xúc sâu sắc, đóng vai trò then chốt trong cốt truyện tổng thể."
            full_content += padding
        
        return full_content
    
    def summarize_script(self, script: str, episode_number: int = 3, script_id: Optional[str] = None) -> Dict:
        """Tóm tắt kịch bản chính với format A4"""
        # Phát hiện thể loại từ categories trong DB
        genre, genre_confidence, category_id = self.detect_genre_from_categories(script)
        
        # Trích xuất cảnh quan trọng
        all_scenes = self.extract_key_scenes(script)
        
        if not all_scenes:
            return {
                'error': 'Không thể trích xuất nội dung từ kịch bản',
                'script_length': len(script),
                'word_count': len(script.split())
            }
        
        # Chia scenes thành các tập
        scenes_per_episode = len(all_scenes) // episode_number
        episodes = []
        
        for ep_num in range(episode_number):
            start_idx = ep_num * scenes_per_episode
            end_idx = start_idx + scenes_per_episode if ep_num < episode_number - 1 else len(all_scenes)
            
            episode_scenes = all_scenes[start_idx:end_idx]
            episode_summary = self.create_a4_episode_summary(episode_scenes, ep_num + 1, genre)
            episodes.append(episode_summary)
        
        # Tạo tóm tắt tổng quan
        overall_summary = self._create_overall_summary(script, genre, all_scenes)
        
        result = {
            'overall_summary': overall_summary,
            'genre': genre,
            'genre_confidence': genre_confidence,
            'category_id': str(category_id) if category_id else None,
            'total_episodes': episode_number,
            'episodes': episodes,
            'created_at': datetime.now(),
            'script_stats': {
                'original_length': len(script),
                'word_count': len(script.split()),
                'sentence_count': len(sent_tokenize(script)),
                'total_scenes_extracted': len(all_scenes)
            }
        }
        
        # Lưu vào MongoDB
        self._save_to_database(script, result, script_id)
        
        return result
    
    def _create_overall_summary(self, text: str, genre: str, scenes: List[Dict]) -> str:
        """Tạo tóm tắt tổng quan chi tiết"""
        # Phân tích nhân vật chính
        characters = self._extract_main_characters(text)
        
        # Phân tích xung đột chính
        main_conflicts = self._extract_main_conflicts(scenes)
        
        # Phân tích theme chính
        themes = self._extract_themes(text, genre)
        
        summary_parts = [
            f"🎭 Thể loại: {genre.title()}",
            f"👥 Nhân vật chính: {', '.join(characters[:3]) if characters else 'Đang phân tích'}",
            f"⚔️ Xung đột chính: {main_conflicts[0] if main_conflicts else 'Xung đột nội tâm và hoàn cảnh'}",
            f"🎯 Chủ đề: {', '.join(themes[:2]) if themes else 'Tình yêu và cuộc sống'}",
            f"📊 Cấu trúc: Chia thành {len(scenes)} cảnh chính với nhiều tình tiết hấp dẫn"
        ]
        
        return '\n'.join(summary_parts)
    
    def _extract_main_characters(self, text: str) -> List[str]:
        """Trích xuất nhân vật chính thông minh hơn"""
        # Tìm tên riêng (từ viết hoa)
        potential_names = re.findall(r'\b[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+', text)
        
        # Lọc bỏ từ thông thường
        common_words = ['Tôi', 'Anh', 'Chị', 'Em', 'Ông', 'Bà', 'Cô', 'Chú', 'Thầy', 'Cô', 'Một', 'Hai', 'Ba']
        
        # Đếm tần suất xuất hiện
        name_counts = Counter([name for name in potential_names 
                              if name not in common_words and len(name) > 2])
        
        # Lấy những tên xuất hiện nhiều nhất
        main_characters = [name for name, count in name_counts.most_common(5) if count >= 2]
        
        return main_characters
    
    def _extract_main_conflicts(self, scenes: List[Dict]) -> List[str]:
        """Trích xuất xung đột chính"""
        conflict_keywords = [
            'xung đột', 'tranh cãi', 'đối đầu', 'chống lại', 'bất đồng',
            'cãi nhau', 'phản đối', 'không đồng ý', 'bất hòa', 'mâu thuẫn'
        ]
        
        conflicts = []
        for scene in scenes:
            text = scene['text'].lower()
            for keyword in conflict_keywords:
                if keyword in text:
                    # Lấy câu chứa từ khóa xung đột
                    sentences = scene['text'].split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            conflicts.append(sentence.strip()[:100] + '...')
                            break
                    break
        
        return conflicts[:3]
    
    def _extract_themes(self, text: str, genre: str) -> List[str]:
        """Trích xuất chủ đề chính"""
        theme_keywords = {
            'tình yêu': ['yêu', 'tình yêu', 'lãng mạn', 'hẹn hò', 'cưới'],
            'gia đình': ['gia đình', 'bố mẹ', 'con cái', 'anh em', 'họ hàng'],
            'tình bạn': ['bạn bè', 'tình bạn', 'đồng nghiệp', 'cùng nhau'],
            'sự nghiệp': ['công việc', 'sự nghiệp', 'thành công', 'thất bại', 'cố gắng'],
            'công lý': ['công lý', 'đúng sai', 'thiện ác', 'chính nghĩa'],
            'phiêu lưu': ['phiêu lưu', 'khám phá', 'mạo hiểm', 'du lịch'],
            'hài hước': ['vui nhộn', 'hài hước', 'cười', 'giải trí']
        }
        
        text_lower = text.lower()
        theme_scores = {}
        
        for theme, keywords in theme_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                theme_scores[theme] = score
        
        # Sắp xếp theo điểm số
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [theme for theme, score in sorted_themes]
    
    def _save_to_database(self, original_script: str, result: Dict, script_id: Optional[str] = None):
        """Lưu kết quả vào MongoDB"""
        try:
            # Tạo document cho AI summary
            ai_summary_doc = {
                'script_id': ObjectId(script_id) if script_id else None,
                'original_script': original_script,
                'summary_result': result,
                'genre': result['genre'],
                'genre_confidence': result['genre_confidence'],
                'category_id': ObjectId(result['category_id']) if result.get('category_id') else None,
                'total_episodes': result['total_episodes'],
                'quality_score': self._calculate_quality_score(result),
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'feedback_count': 0,
                'average_rating': 0.0,
                'ai_version': '1.0.0',
                'processing_stats': {
                    'scenes_extracted': result['script_stats']['total_scenes_extracted'],
                    'original_word_count': result['script_stats']['word_count'],
                    'summary_word_count': sum(ep['word_count'] for ep in result['episodes'])
                }
            }
            
            inserted = self.ai_summaries_collection.insert_one(ai_summary_doc)
            result['ai_summary_id'] = str(inserted.inserted_id)
            print(f"✅ Saved AI summary to database with ID: {inserted.inserted_id}")
            
        except Exception as e:
            print(f"❌ Error saving to database: {e}")
    def get_all_summaries(self, limit: int = 100):
        """Lấy danh sách các AI summary"""
        try:
            summaries_cursor = self.ai_summaries_collection.find().sort("created_at", -1).limit(limit)
            summaries = []
            for doc in summaries_cursor:
                summaries.append({
                    "id": str(doc["_id"]),
                    "script_id": str(doc["script_id"]) if doc["script_id"] else None,
                    "genre": doc.get("genre"),
                    "genre_confidence": doc.get("genre_confidence"),
                    "total_episodes": doc.get("total_episodes"),
                    "quality_score": doc.get("quality_score"),
                    "created_at": doc.get("created_at"),
                    "average_rating": doc.get("average_rating"),
                    "feedback_count": doc.get("feedback_count")
                })
            return summaries
        except Exception as e:
            print(f"❌ Error fetching summaries: {e}")
            raise
    
    def _calculate_quality_score(self, result: Dict) -> float:
        """Tính điểm chất lượng tóm tắt"""
        score = 0.5  # Base score
        
        # Điểm dựa trên genre confidence
        score += result['genre_confidence'] * 0.2
        
        # Điểm dựa trên số lượng episodes
        if 2 <= result['total_episodes'] <= 10:
            score += 0.1
        
        # Điểm dựa trên độ dài tóm tắt
        for episode in result['episodes']:
            word_count = episode['word_count']
            if 550 <= word_count <= 650:  # Đúng độ dài A4
                score += 0.05
        
        # Điểm dựa trên số cảnh được trích xuất
        scenes_count = result['script_stats']['total_scenes_extracted']
        if scenes_count >= 10:
            score += 0.1
        
        return min(score, 1.0)
    
    def get_feedbacks_for_training(self, limit: int = 100) -> List[Dict]:
        """Lấy feedback từ DB để training"""
        try:
            # Lấy feedback có rating và content
            feedbacks = list(self.feedbacks_collection.find(
                {
                    "isVisible": True,
                    "isFlagged": False,
                    "content": {"$exists": True, "$ne": ""},
                    "rate": {"$exists": True}
                },
                {
                    "content": 1,
                    "rate": 1,
                    "movieId": 1,
                    "storyId": 1,
                    "createdAt": 1,
                    "helpfulCount": 1
                }
            ).sort("createdAt", -1).limit(limit))
            
            return feedbacks
        except Exception as e:
            print(f"❌ Error fetching feedbacks: {e}")
            return []
    
    def train_from_feedback(self, feedback_list: List[str] = None) -> Dict:
        """Huấn luyện model từ feedback"""
        try:
            # Nếu không có feedback_list, lấy từ DB
            if not feedback_list:
                db_feedbacks = self.get_feedbacks_for_training()
                feedback_list = [fb['content'] for fb in db_feedbacks if fb.get('content')]
            
            if not feedback_list:
                return {'error': 'Không có feedback để huấn luyện'}
            
            # Phân tích feedback và cập nhật model
            training_result = self._process_feedback_for_training(feedback_list)
            
            # Lưu training metrics
            training_metrics = {
                'feedback_count': len(feedback_list),
                'positive_feedback': sum(1 for f in feedback_list if self._analyze_feedback_sentiment(f) == 'positive'),
                'negative_feedback': sum(1 for f in feedback_list if self._analyze_feedback_sentiment(f) == 'negative'),
                'training_improvements': training_result,
                'trained_at': datetime.now(),
                'ai_version': '1.0.0'
            }
            
            self.training_metrics_collection.insert_one(training_metrics)
            
            return {
                'success': True,
                'feedback_processed': len(feedback_list),
                'improvements': training_result,
                'training_id': str(training_metrics.get('_id'))
            }
            
        except Exception as e:
            return {'error': f'Lỗi trong quá trình huấn luyện: {str(e)}'}
    
    def _analyze_feedback_sentiment(self, feedback: str) -> str:
        """Phân tích sentiment của feedback"""
        positive_words = ['tốt', 'hay', 'xuất sắc', 'tuyệt vời', 'ổn', 'đúng', 'chính xác', 'hài lòng', 'thích', 'yêu']
        negative_words = ['tệ', 'sai', 'không tốt', 'kém', 'thiếu', 'không đúng', 'không hay', 'dở', 'chán']
        
        feedback_lower = feedback.lower()
        positive_count = sum(1 for word in positive_words if word in feedback_lower)
        negative_count = sum(1 for word in negative_words if word in feedback_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _process_feedback_for_training(self, feedback_list: List[str]) -> Dict:
        """Xử lý feedback để cải thiện model"""
        improvements = {
            'genre_detection': 0,
            'scene_extraction': 0,
            'summary_quality': 0,
            'episode_structure': 0,
            'character_analysis': 0
        }
        
        for feedback in feedback_list:
            feedback_lower = feedback.lower()
            
            # Cải thiện genre detection
            if any(word in feedback_lower for word in ['thể loại', 'genre', 'sai thể loại', 'không đúng thể loại']):
                improvements['genre_detection'] += 1
            
            # Cải thiện scene extraction
            if any(word in feedback_lower for word in ['cảnh', 'thiếu', 'bỏ sót', 'không đủ', 'cần thêm']):
                improvements['scene_extraction'] += 1
            
            # Cải thiện summary quality
            if any(word in feedback_lower for word in ['tóm tắt', 'chất lượng', 'chi tiết', 'ngắn gọn']):
                improvements['summary_quality'] += 1
            
            # Cải thiện episode structure
            if any(word in feedback_lower for word in ['tập', 'cấu trúc', 'chia', 'phần']):
                improvements['episode_structure'] += 1
                
            # Cải thiện character analysis
            if any(word in feedback_lower for word in ['nhân vật', 'character', 'vai diễn', 'diễn viên']):
                improvements['character_analysis'] += 1
        
        # Lưu learned patterns
        self._save_learned_patterns(improvements)
        
        return improvements
    
    def _save_learned_patterns(self, improvements: Dict):
        """Lưu patterns đã học"""
        pattern_doc = {
            'improvements': improvements,
            'learned_at': datetime.now(),
            'pattern_type': 'feedback_analysis',
            'ai_version': '1.0.0'
        }
        self.learned_patterns_collection.insert_one(pattern_doc)
    
    def _load_models(self):
        """Load models đã được train trước đó"""
        try:
            # Tạo sample data để train model ban đầu
            self._initialize_models_with_sample_data()
            print("✅ Models initialized successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load pre-trained models: {e}")
    
    def _initialize_models_with_sample_data(self):
        """Khởi tạo models với sample data"""
        # Sample data cho genre classification
        sample_texts = [
            "Anh yêu em nhiều lắm. Chúng ta sẽ cưới nhau.",
            "Súng nổ vang. Anh ta lao vào cuộc chiến đấu.",
            "Cười lớn. Tình huống này thật buồn cười.",
            "Gia đình tôi rất hạnh phúc. Tình cảm sâu sắc.",
            "Bí ẩn đằng sau cái chết. Sợ hãi bao trùm."
        ]
        
        sample_labels = ['romance', 'action', 'comedy', 'drama', 'thriller']
        
        try:
            # Fit vectorizer với sample data
            self.vectorizer.fit(sample_texts)
            
            # Train genre classifier với sample data
            X_sample = self.vectorizer.transform(sample_texts)
            self.genre_classifier.fit(X_sample, sample_labels)
            
        except Exception as e:
            print(f"⚠️ Error initializing models: {e}")

    def get_training_statistics(self) -> Dict:
        """Lấy thống kê training"""
        try:
            # Thống kê từ AI summaries
            total_summaries = self.ai_summaries_collection.count_documents({})
            
            # Thống kê từ feedbacks
            total_feedback = self.feedbacks_collection.count_documents({
                "isVisible": True,
                "isFlagged": False
            })
            
            # Feedback gần đây (7 ngày)
            week_ago = datetime.now() - timedelta(days=7)
            recent_feedback = self.feedbacks_collection.count_documents({
                'createdAt': {'$gte': week_ago},
                "isVisible": True,
                "isFlagged": False
            })
            
            # Average quality score từ AI summaries
            pipeline = [
                {'$group': {'_id': None, 'avg_quality': {'$avg': '$quality_score'}}}
            ]
            avg_quality_result = list(self.ai_summaries_collection.aggregate(pipeline))
            avg_quality = avg_quality_result[0]['avg_quality'] if avg_quality_result else 0
            
            # Average rating từ feedbacks
            rating_pipeline = [
                {'$match': {'isVisible': True, 'isFlagged': False}},
                {'$group': {'_id': None, 'avg_rating': {'$avg': '$rate'}}}
            ]
            avg_rating_result = list(self.feedbacks_collection.aggregate(rating_pipeline))
            avg_rating = avg_rating_result[0]['avg_rating'] if avg_rating_result else 0
            
            # Training metrics
            training_count = self.training_metrics_collection.count_documents({})
            
            return {
                'total_summaries_generated': total_summaries,
                'total_feedback_received': total_feedback,
                'recent_feedback_count': recent_feedback,
                'average_quality_score': round(avg_quality, 2),
                'average_user_rating': round(avg_rating, 2),
                'total_training_sessions': training_count,
                'last_updated': datetime.now(),
                'ai_version': '1.0.0'
            }
            
        except Exception as e:
            return {'error': f'Lỗi lấy thống kê: {str(e)}'}

    def get_script_by_id(self, script_id: str) -> Optional[Dict]:
        """Lấy script từ DB theo ID"""
        try:
            script = self.scripts_collection.find_one({"_id": ObjectId(script_id)})
            if script:
                script['_id'] = str(script['_id'])
                if script.get('movieId'):
                    script['movieId'] = str(script['movieId'])
            return script
        except Exception as e:
            print(f"❌ Error fetching script: {e}")
            return None

    def update_summary_feedback(self, ai_summary_id: str, feedback_rating: float, feedback_text: str = ""):
        """Cập nhật feedback cho AI summary"""
        try:
            # Cập nhật AI summary với feedback
            update_data = {
                'feedback_count': {'$inc': 1},
                'last_feedback_at': datetime.now()
            }
            
            # Tính lại average rating
            existing_summary = self.ai_summaries_collection.find_one({"_id": ObjectId(ai_summary_id)})
            if existing_summary:
                current_avg = existing_summary.get('average_rating', 0.0)
                current_count = existing_summary.get('feedback_count', 0)
                new_avg = ((current_avg * current_count) + feedback_rating) / (current_count + 1)
                update_data['average_rating'] = round(new_avg, 2)
            
            self.ai_summaries_collection.update_one(
                {"_id": ObjectId(ai_summary_id)},
                {"$set": update_data}
            )
            
            print(f"✅ Updated feedback for AI summary: {ai_summary_id}")
            
        except Exception as e:
            print(f"❌ Error updating summary feedback: {e}")

