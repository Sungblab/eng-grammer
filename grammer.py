import spacy
from transformers import MarianMTModel, MarianTokenizer
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalGrammarAnalyzer:
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("Spacy 모델 로드 성공.")
        except OSError:
            logger.error("Spacy 모델 'en_core_web_sm'을 찾을 수 없습니다. 모델을 다운로드하세요.")
            raise

    def _analyze_tense(self, doc):
        """더욱 정교한 시제 분석"""
        aux_verbs = []
        main_verbs = []
        all_verb_tokens = []
        
        has_ing_form = False
        has_past_participle = False
        has_been = False
        
        for token in doc:
            if token.pos_ in ["VERB", "AUX"]:
                all_verb_tokens.append(token)
                
                if token.tag_ == "VBG":
                    has_ing_form = True
                elif token.tag_ == "VBN":
                    has_past_participle = True
                    if token.text.lower() == "been":
                        has_been = True
                
                if token.text.lower() in ["'s"]:
                    next_token = doc[token.i + 1] if token.i + 1 < len(doc) else None
                    if next_token and next_token.tag_ == "VBG":
                        aux_verbs.append(("is", "VBZ"))
                        has_ing_form = True
                    elif next_token and next_token.text.lower() == "been":
                        aux_verbs.append(("has", "VBZ"))
                    else:
                        aux_verbs.append(("has", "VBZ"))
                elif token.text.lower() in ["'ve"]:
                    aux_verbs.append(("have", "VBP"))
                elif token.text.lower() in ["'m"]:
                    aux_verbs.append(("am", "VBP"))
                elif token.text.lower() in ["'re"]:
                    aux_verbs.append(("are", "VBP"))
                elif token.dep_ == "aux":
                    aux_verbs.append((token.text.lower(), token.tag_))
                else:
                    main_verbs.append((token.text.lower(), token.tag_))
        
        # 시제 분석
        if has_been and has_ing_form:
            if any(verb in ["have", "has", "'ve", "'s"] for verb, _ in aux_verbs):
                return "현재 완료 진행형"
            if any(verb == "had" for verb, _ in aux_verbs):
                return "과거 완료 진행형"
        
        if has_ing_form:
            if any(verb in ["am", "is", "are", "'m", "'s", "'re"] for verb, _ in aux_verbs):
                return "현재 진행형"
            if any(verb in ["was", "were"] for verb, _ in aux_verbs):
                return "과거 진행형"
        
        if has_past_participle:
            if any(verb in ["have", "has", "'ve", "'s"] for verb, _ in aux_verbs):
                return "현재 완료형"
            if any(verb == "had" for verb, _ in aux_verbs):
                return "과거 완료형"
        
        if has_past_participle and not has_been:
            if any(verb in ["am", "is", "are"] for verb, _ in aux_verbs):
                return "현재 수동태"
            if any(verb in ["was", "were"] for verb, _ in aux_verbs):
                return "과거 수동태"
        
        if any(token.tag_ == "VBD" for token in all_verb_tokens):
            return "단순 과거형"
        
        if any(token.tag_ in ["VBP", "VBZ"] for token in all_verb_tokens):
            return "단순 현재형"
        
        return "알 수 없음"

    def _analyze_voice(self, doc):
        """더욱 정확한 태 분석"""
        verb_pattern = [(token.text.lower(), token.tag_) for token in doc if token.pos_ in ["VERB", "AUX"]]
        
        be_verbs = ["am", "is", "are", "was", "were", "be", "been"]
        has_be_verb = any(word in be_verbs for word, _ in verb_pattern)
        has_past_participle = any(tag == "VBN" for _, tag in verb_pattern)
        
        if has_be_verb and has_past_participle and not any(word in ["have", "has", "had"] for word, _ in verb_pattern):
            return "수동태"
                
        return "능동태"

    def _analyze_structure(self, doc):
        """개선된 문장 구조 분석"""
        subjects = [token for token in doc if "subj" in token.dep_]
        verbs = [token for token in doc if token.pos_ in ["VERB", "AUX"]]
        objects = [token for token in doc if "obj" in token.dep_]
        
        if subjects and verbs:
            if objects:
                return "SVO"
            return "SV"
                
        return "기타"

    def _check_article_usage(self, doc):
        """개선된 관사 사용 검사"""
        for token in doc:
            if token.pos_ == "NOUN" and token.pos_ != "PROPN":
                if token.text.lower() in [
                    "school", "home", "work", "breakfast", "lunch", "dinner", 
                    "yesterday", "today", "tomorrow", "tv", "television"
                ]:
                    continue
                if any(child.dep_ in ["poss", "nummod"] for child in token.children):
                    continue
                if token.dep_ == "pobj":
                    continue
                prev_token = doc[token.i - 1] if token.i > 0 else None
                if not prev_token or prev_token.pos_ not in ["DET", "PRON", "ADJ", "PROPN"]:
                    return False
        return True

    def analyze_grammar(self, text):
        """문법 분석 실행"""
        doc = self.nlp(text)
        
        analysis = {
            '시제': self._analyze_tense(doc),
            '태': self._analyze_voice(doc),
            '문장 구조': self._analyze_structure(doc),
            '주어-동사 일치': self._check_subject_verb_agreement(doc),
            '관사 사용': self._check_article_usage(doc)
        }
        
        return analysis

    def _check_subject_verb_agreement(self, doc):
        """주어와 동사의 일치 여부 확인"""
        for token in doc:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                if not token.morph.get("Number") or not token.head.morph.get("Number"):
                    continue
                if token.morph.get("Number") != token.head.morph.get("Number"):
                    return False
        return True

# 테스트
if __name__ == "__main__":
    analyzer = FinalGrammarAnalyzer()
    
    test_sentences = [
        "He is writing a letter.",          # 현재 진행형
        "They've finished the project.",    # 현재 완료형
        "We were watching TV.",             # 과거 진행형
        "The food was being cooked.",       # 과거 수동태
        "The window broke."                 # 단순 과거형
    ]
    
    for sentence in test_sentences:
        analysis = analyzer.analyze_grammar(sentence)
        print(f"\n문장: {sentence}")
        print("분석 결과:", analysis)   