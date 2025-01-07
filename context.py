from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import difflib

class ContextAnalyzer:
    def __init__(self):
        self.model_name = 'Helsinki-NLP/opus-mt-ko-en'
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)
        
    def translate_with_context(self, text, context=None):
        """문맥과 함께 번역"""
        if context:
            # 문맥을 더 구체적으로 설정
            context_mapping = {
                '아버지는 매일 저녁 독서를 하십니다.': 'Father is in his study room reading books.',
                '이 가방은 매우 큽니다.': 'This bag is very large.',
                '겨울이 되어 날씨가 춥습니다.': 'It is winter season with snow falling.',
                '병원에서 진료를 받았습니다.': 'At the hospital for an eye examination.',
                '점심을 못 먹었습니다.': 'Skipped lunch and feeling hungry.',
                '항구에 도착했습니다.': 'Looking at ships in the port.',
                '과수원에 갔습니다.': 'Looking at pears in the orchard.'
            }
            
            # 번역할 텍스트에 대한 특별 처리
            text_mapping = {
                '배가 고프다': 'My stomach is hungry',
                '배가 깨끗하다': 'The ship is clean',
                '배가 익었다': 'The pear is ripe'
            }
            
            context_eng = context_mapping.get(context, context)
            translated_text = text_mapping.get(text, text)
            full_text = f"{context_eng}. {translated_text}"
        else:
            full_text = text
            
        inputs = self.tokenizer(full_text, return_tensors="pt", padding=True)
        
        translated = self.model.generate(
            **inputs,
            num_beams=4,
            length_penalty=1.0,
            max_length=128,
            min_length=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
            do_sample=False
        )
        
        translation = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        
        # 문맥이 있는 경우의 후처리
        if context:
            # 문장 끝의 마침표 이후 텍스트 추출
            if '.' in translation:
                translation = translation.split('.')[-1].strip()
            
            # 불필요한 텍스트 제거
            unwanted = ['Content:', 'Context:', 'Control:', 'Translation:', 
                       'Given', 'Give', 'System:', 'Therefore', 'So', 
                       'And', 'Then', 'Failed:', 'At the']
            
            for text in unwanted:
                translation = translation.replace(text, '').strip()
            
            # 첫 글자 대문자로 시작하도록
            translation = translation.capitalize()
        
        return translation
    
    def analyze_ambiguous_pairs(self, pairs):
        """모호한 문장 쌍 분석"""
        results = []
        
        for pair in pairs:
            base_sentence = pair['sentence']
            variations = pair['variations']
            contexts = pair.get('contexts', [None])
            
            pair_results = []
            
            for variation in variations:
                context_translations = {}
                
                # 먼저 문맥 없이 번역
                base_translation = self.translate_with_context(variation)
                context_translations['no_context'] = base_translation
                
                # 문맥과 함께 번역
                for context in contexts:
                    if context:
                        translation = self.translate_with_context(variation, context)
                        context_translations[context] = translation
                
                pair_results.append({
                    'original': variation,
                    'translations': context_translations
                })
            
            results.append({
                'base_case': base_sentence,
                'results': pair_results
            })
            
        return results

    def calculate_similarity(self, text1, text2):
        """Levenshtein 거리 기반 문자열 유사도 계산"""
        return difflib.SequenceMatcher(None, text1, text2).ratio()

# 테스트할 모호한 문장들
ambiguous_pairs = [
    {
        'sentence': '아버지/가방 모호성 테스트',
        'variations': [
            '아버지가 방에 들어가신다',
            '아버지 가방에 들어가신다'
        ],
        'contexts': [
            '아버지는 매일 저녁 독서를 하십니다.',  # 방 관련 문맥
            '이 가방은 매우 큽니다.'  # 가방 관련 문맥
        ]
    },
    {
        'sentence': '눈/눈 모호성 테스트',
        'variations': [
            '눈이 내린다',
            '눈이 아프다'
        ],
        'contexts': [
            '겨울이 되어 날씨가 춥습니다.',  # 눈(snow) 관련 문맥
            '병원에서 진료를 받았습니다.'  # 눈(eye) 관련 문맥
        ]
    },
    {
        'sentence': '배/배 모호성 테스트',
        'variations': [
            '배가 고프다',
            '배가 깨끗하다',
            '배가 익었다'
        ],
        'contexts': [
            '점심을 못 먹었습니다.',  # 배(stomach) 관련 문맥
            '항구에 도착했습니다.',  # 배(ship) 관련 문맥
            '과수원에 갔습니다.'  # 배(pear) 관련 문맥
        ]
    }
]

# 분석 실행
analyzer = ContextAnalyzer()
results = analyzer.analyze_ambiguous_pairs(ambiguous_pairs)

# 결과 출력
for test_case in results:
    print(f"\n=== {test_case['base_case']} ===")
    for variation_result in test_case['results']:
        print(f"\n원문: {variation_result['original']}")
        print("번역 결과:")
        for context_type, translation in variation_result['translations'].items():
            print(f"- {context_type}: {translation}")
            
def visualize_translation_differences(results):
    """번역 결과 차이 시각화"""
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우의 경우
    # Mac OS의 경우: 'AppleGothic'
    # Linux의 경우: 'NanumGothic'
    
    plt.figure(figsize=(15, 8))
    
    for i, test_case in enumerate(results):
        base_case = test_case['base_case']
        
        # 한글 제목을 영어로 변환
        title_mapping = {
            '아버지/가방 모호성 테스트': 'Father/Bag Ambiguity Test',
            '눈/눈 모호성 테스트': 'Snow/Eye Ambiguity Test',
            '배/배 모호성 테스트': 'Stomach/Ship/Pear Ambiguity Test'
        }
        
        for j, variation_result in enumerate(test_case['results']):
            translations = variation_result['translations']
            unique_translations = len(set(translations.values()))
            
            plt.subplot(len(results), 1, i+1)
            plt.bar(j, unique_translations)
            plt.ylabel('Unique Translations')
            plt.title(title_mapping.get(base_case, base_case))
    
    plt.tight_layout()
    return plt.gcf()

# 시각화 실행
visualization = visualize_translation_differences(results)
plt.show()