from __future__ import unicode_literals
import tweepy
import numpy as np
import pickle
import pandas as pd
import emoji
from nltk.corpus import stopwords
from textblob import TextBlob
import re
from tashaphyne.stemming import ArabicLightStemmer
import nltk
import mtranslate
import pyarabic.araby as araby
from aiogoogletrans import Translator
import asyncio
import multiprocessing as mp
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager
import swifter
from joblib import Parallel, delayed

nltk.download('stopwords')
stops = set(stopwords.words("arabic"))
stop_word_comp = {"،", "آض", "آمينَ", "آه", "آهاً", "آي", "أ", "أب", "أجل", "أجمع", "أخ", "أخذ", "أصبح", "أضحى", "أقبل",
                  "أقل", "أكثر", "ألا", "أم", "أما", "أمامك", "أمامكَ", "أمسى", "أمّا", "أن", "أنا", "أنت", "أنتم",
                  "أنتما", "أنتن", "أنتِ", "أنشأ", "أنّى", "أو", "أوشك", "أولئك", "أولئكم", "أولاء", "أولالك", "أوّهْ",
                  "أي", "أيا", "أين", "أينما", "أيّ", "أَنَّ", "أََيُّ", "أُفٍّ", "إذ", "إذا", "إذاً", "إذما", "إذن",
                  "إلى", "إليكم", "إليكما", "إليكنّ", "إليكَ", "إلَيْكَ", "إلّا", "إمّا", "إن", "إنّما", "إي", "إياك",
                  "إياكم", "إياكما", "إياكن", "إيانا", "إياه", "إياها", "إياهم", "إياهما", "إياهن", "إياي", "إيهٍ",
                  "إِنَّ", "ا", "ابتدأ", "اثر", "اجل", "احد", "اخرى", "اخلولق", "اذا", "اربعة", "ارتدّ", "استحال",
                  "اطار", "اعادة", "اعلنت", "اف", "اكثر", "اكد", "الألاء", "الألى", "الا", "الاخيرة", "الان", "الاول",
                  "الاولى", "التى", "التي", "الثاني", "الثانية", "الذاتي", "الذى", "الذي", "الذين", "السابق", "الف",
                  "اللائي", "اللاتي", "اللتان", "اللتيا", "اللتين", "اللذان", "اللذين", "اللواتي", "الماضي", "المقبل",
                  "الوقت", "الى", "اليوم", "اما", "امام", "امس", "ان", "انبرى", "انقلب", "انه", "انها", "او", "اول",
                  "اي", "ايار", "ايام", "ايضا", "ب", "بات", "باسم", "بان", "بخٍ", "برس", "بسبب", "بسّ", "بشكل", "بضع",
                  "بطآن", "بعد", "بعض", "بك", "بكم", "بكما", "بكن", "بل", "بلى", "بما", "بماذا", "بمن", "بن", "بنا",
                  "به", "بها", "بي", "بيد", "بين", "بَسْ", "بَلْهَ", "بِئْسَ", "تانِ", "تانِك", "تبدّل", "تجاه",
                  "تحوّل", "تلقاء", "تلك", "تلكم", "تلكما", "تم", "تينك", "تَيْنِ", "تِه", "تِي", "ثلاثة", "ثم", "ثمّ",
                  "ثمّة", "ثُمَّ", "جعل", "جلل", "جميع", "جير", "حار", "حاشا", "حاليا", "حاي", "حتى", "حرى", "حسب",
                  "حم", "حوالى", "حول", "حيث", "حيثما", "حين", "حيَّ", "حَبَّذَا", "حَتَّى", "حَذارِ", "خلا", "خلال",
                  "دون", "دونك", "ذا", "ذات", "ذاك", "ذانك", "ذانِ", "ذلك", "ذلكم", "ذلكما", "ذلكن", "ذو", "ذوا",
                  "ذواتا", "ذواتي", "ذيت", "ذينك", "ذَيْنِ", "ذِه", "ذِي", "راح", "رجع", "رويدك", "ريث", "رُبَّ",
                  "زيارة", "سبحان", "سرعان", "سنة", "سنوات", "سوف", "سوى", "سَاءَ", "سَاءَمَا", "شبه", "شخصا", "شرع",
                  "شَتَّانَ", "صار", "صباح", "صفر", "صهٍ", "صهْ", "ضد", "ضمن", "طاق", "طالما", "طفق", "طَق", "ظلّ",
                  "عاد", "عام", "عاما", "عامة", "عدا", "عدة", "عدد", "عدم", "عسى", "عشر", "عشرة", "علق", "على", "عليك",
                  "عليه", "عليها", "علًّ", "عن", "عند", "عندما", "عوض", "عين", "عَدَسْ", "عَمَّا", "غدا", "غير", "ـ",
                  "ف", "فان", "فلان", "فو", "فى", "في", "فيم", "فيما", "فيه", "فيها", "قال", "قام", "قبل", "قد", "قطّ",
                  "قلما", "قوة", "كأنّما", "كأين", "كأيّ", "كأيّن", "كاد", "كان", "كانت", "كذا", "كذلك", "كرب", "كل",
                  "كلا", "كلاهما", "كلتا", "كلم", "كليكما", "كليهما", "كلّما", "كلَّا", "كم", "كما", "كي", "كيت", "كيف",
                  "كيفما", "كَأَنَّ", "كِخ", "لئن", "لا", "لات", "لاسيما", "لدن", "لدى", "لعمر", "لقاء", "لك", "لكم",
                  "لكما", "لكن", "لكنَّما", "لكي", "لكيلا", "للامم", "لم", "لما", "لمّا", "لن", "لنا", "له", "لها",
                  "لو", "لوكالة", "لولا", "لوما", "لي", "لَسْتَ", "لَسْتُ", "لَسْتُم", "لَسْتُمَا", "لَسْتُنَّ",
                  "لَسْتِ", "لَسْنَ", "لَعَلَّ", "لَكِنَّ", "لَيْتَ", "لَيْسَ", "لَيْسَا", "لَيْسَتَا", "لَيْسَتْ",
                  "لَيْسُوا", "لَِسْنَا", "ما", "ماانفك", "مابرح", "مادام", "ماذا", "مازال", "مافتئ", "مايو", "متى",
                  "مثل", "مذ", "مساء", "مع", "معاذ", "مقابل", "مكانكم", "مكانكما", "مكانكنّ", "مكانَك", "مليار",
                  "مليون", "مما", "ممن", "من", "منذ", "منها", "مه", "مهما", "مَنْ", "مِن", "نحن", "نحو", "نعم", "نفس",
                  "نفسه", "نهاية", "نَخْ", "نِعِمّا", "نِعْمَ", "ها", "هاؤم", "هاكَ", "هاهنا", "هبّ", "هذا", "هذه",
                  "هكذا", "هل", "هلمَّ", "هلّا", "هم", "هما", "هن", "هنا", "هناك", "هنالك", "هو", "هي", "هيا", "هيت",
                  "هيّا", "هَؤلاء", "هَاتانِ", "هَاتَيْنِ", "هَاتِه", "هَاتِي", "هَجْ", "هَذا", "هَذانِ", "هَذَيْنِ",
                  "هَذِه", "هَذِي", "هَيْهَاتَ", "و", "و6", "وا", "واحد", "واضاف", "واضافت", "واكد", "وان", "واهاً",
                  "واوضح", "وراءَك", "وفي", "وقال", "وقالت", "وقد", "وقف", "وكان", "وكانت", "ولا", "ولم", "ومن", "مَن",
                  "وهو", "وهي", "ويكأنّ", "وَيْ", "وُشْكَانََ", "يكون", "يمكن", "يوم", "ّأيّان"}
ArListem = ArabicLightStemmer()
translator = Translator()
loop = asyncio.get_event_loop()


class get_auth():
    def __init__(self):
        self.consumer_key = "RsQwvzkWoh9v2vjzWyREUaHtP"
        self.consumer_secret = "OcZ33vRYZmxdC7AKu49gkfVITzwMHA4mRW4EqYwopQ0SiPi4se"
        self.auth1 = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret, "/")
        self.api = tweepy.API(self.auth1)
        self.redirect_url = self.auth1.get_authorization_url()
        self.searches = []
        self.screen_name = None

    def regenerate(self):
        self.auth1 = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret, "/")
        self.redirect_url = self.auth1.get_authorization_url()
        self.api = tweepy.API(self.auth1)

    def getdata(self, username):
        data = []
        for i in tweepy.Cursor(self.api.user_timeline, screen_name=username, tweet_mode='extended', include_rts=True,
                               exclude_replies=True).items(200):
            text = i._json['full_text']
            date = int(i._json['created_at'][-4:])
            lang = i._json['lang']
            data.append([username, text, lang, date])
        data = pd.DataFrame(data, columns=['username', 'tweet', 'lang', 'date'])
        return data


class Trait():
    def __init__(self):
        with open('final_ml_models/O.pkl', 'rb') as f:
            self.modelo = pickle.load(f)
        with open('final_ml_models/C.pkl', 'rb') as f:
            self.modelc = pickle.load(f)
        with open('final_ml_models/E.pkl', 'rb') as f:
            self.modele = pickle.load(f)
        with open('final_ml_models/A.pkl', 'rb') as f:
            self.modela = pickle.load(f)
        with open('final_ml_models/N.pkl', 'rb') as f:
            self.modeln = pickle.load(f)
        self.features = self.modela.feature_names_in_

    def preprocessing(self, username, auth):
        self.data = auth.getdata(username)
        df = self.data.copy()
        df = df[(df['lang'] == 'ar') | (df['lang'] == 'en') | (df['lang'] == 'und') | (df['lang'] == 'art') | (
                df['lang'] == 'qht')]

        def avg_word(sentence):
            words = sentence.split()
            if len(words) == 0:
                return 0
            return (sum(len(word) for word in words) / len(words))

        def emoji_counter(sentence):
            return emoji.emoji_count(sentence)

        df['Word_Coun'] = df['tweet'].apply(lambda x: len(str(x).split(" ")))
        df['Char_Count'] = df['tweet'].str.len()
        df['Avg_Char_Per_Word'] = df['tweet'].apply(lambda x: avg_word(x))
        df['Emoji_Count'] = df['tweet'].apply(lambda x: emoji_counter(x))

        def to_arabic(text):
            text = text.split(' ')
            clean = []
            for i in text:
                if re.match(r'^[a-zA-Z]+$', i):
                    translated_text = mtranslate.translate(i, "ar")
                    clean.append(translated_text)
                else:
                    clean.append(i)
            return " ".join(clean)

        def normalizeArabic(text):
            text = text.strip()
            text = re.sub("[إأٱآا]", "ا", text)
            text = re.sub("ى", "ي", text)
            text = re.sub("ؤ", "ء", text)
            text = re.sub("ئ", "ء", text)
            text = re.sub("ة", "ه", text)
            noise = re.compile(""" ّ    | # Tashdid
                                     َ    | # Fatha
                                     ً    | # Tanwin Fath
                                     ُ    | # Damma
                                     ٌ    | # Tanwin Damm
                                     ِ    | # Kasra
                                     ٍ    | # Tanwin Kasr
                                     ْ    | # Sukun
                                     ـ     # Tatwil/Kashida
                                 """, re.VERBOSE)
            text = re.sub(noise, '', text)
            text = re.sub(r'(.)\1+', r"\1\1", text)  # Remove longation
            return araby.strip_tashkeel(text)

        def remove_stop_words(text):
            zen = TextBlob(text)
            words = zen.words
            return " ".join([w for w in words if not w in stops and not w in stop_word_comp and len(w) >= 2])

        def split_hashtag_to_words(tag):
            tag = tag.replace('#', '')
            tags = tag.split('_')
            if len(tags) > 1:
                return tags
            pattern = re.compile(r"[A-Z][a-z]+|\d+|[A-Z]+(?![a-z])")
            return pattern.findall(tag)

        def clean_hashtag(text):
            words = text.split()
            text = list()
            for word in words:
                if is_hashtag(word):
                    text.extend(extract_hashtag(word))
                else:
                    text.append(word)
            return " ".join(text)

        def is_hashtag(word):
            if word.startswith("#"):
                return True
            else:
                return False

        def extract_hashtag(text):

            hash_list = ([re.sub(r"(\W+)$", "", i) for i in text.split() if i.startswith("#")])
            word_list = []
            for word in hash_list:
                word_list.extend(split_hashtag_to_words(word))
            return word_list

        with open(
                "emojis.csv",
                'r', encoding='utf-8') as f:
            lines = f.readlines()
            emojis_ar = {}
            for line in lines:
                line = line.strip('\n').split(';')
                emojis_ar.update({line[0].strip(): line[1].strip()})

        def remove_emoji(text):
            emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       u"\U00002702-\U000027B0"
                                       u"\U000024C2-\U0001F251"
                                       "]+", flags=re.UNICODE)
            text = emoji_pattern.sub(r'', text)
            return text

        def emoji_native_translation(text):
            text = text.lower()
            loves = ["<3", "♥", '❤']
            smilefaces = []
            sadfaces = []
            neutralfaces = []

            eyes = ["8", ":", "=", ";"]
            nose = ["'", "`", "-", r"\\"]
            for e in eyes:
                for n in nose:
                    for s in [")", "d", "]", "}", "p"]:
                        smilefaces.append(e + n + s)
                        smilefaces.append(e + s)
                    for s in ["(", "[", "{"]:
                        sadfaces.append(e + n + s)
                        sadfaces.append(e + s)
                    for s in ["|", "/", r"\\"]:
                        neutralfaces.append(e + n + s)
                        neutralfaces.append(e + s)
                    # reversed
                    for s in ["(", "[", "{"]:
                        smilefaces.append(s + n + e)
                        smilefaces.append(s + e)
                    for s in [")", "]", "}"]:
                        sadfaces.append(s + n + e)
                        sadfaces.append(s + e)
                    for s in ["|", "/", r"\\"]:
                        neutralfaces.append(s + n + e)
                        neutralfaces.append(s + e)

            smilefaces = list(set(smilefaces))
            sadfaces = list(set(sadfaces))
            neutralfaces = list(set(neutralfaces))
            t = []
            for w in text.split():
                if w in loves:
                    t.append("حب")
                elif w in smilefaces:
                    t.append("مضحك")
                elif w in neutralfaces:
                    t.append("عادي")
                elif w in sadfaces:
                    t.append("محزن")
                else:
                    t.append(w)
            newText = " ".join(t)
            return newText

        def is_emoji(word):
            if word in emojis_ar:
                return True
            else:
                return False

        def add_space(text):
            return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()

        def translate_emojis(words):
            word_list = list()
            words_to_translate = list()
            for word in words:
                t = emojis_ar.get(word.get('emoji'), None)
                if t is None:
                    word.update({'translation': 'عادي', 'translated': True})
                    # words_to_translate.append('normal')
                else:
                    word.update({'translated': False, 'translation': t})
                    words_to_translate.append(t.replace(':', '').replace('_', ' '))
                word_list.append(word)
            return word_list

        def emoji_unicode_translation(text):
            text = add_space(text)
            words = text.split()
            text_list = list()
            emojis_list = list()
            c = 0
            for word in words:
                if is_emoji(word):
                    emojis_list.append({'emoji': word, 'emplacement': c})
                else:
                    text_list.append(word)
                c += 1
            emojis_translated = translate_emojis(emojis_list)
            for em in emojis_translated:
                text_list.insert(em.get('emplacement'), em.get('translation'))
            text = " ".join(text_list)
            return text

        def clean_emoji(text):
            text = emoji_native_translation(text)
            text = emoji_unicode_translation(text)
            return text

        def clean_tweet(text):
            text = re.sub('#\d+K\d+', ' ', text)  # years like 2K19
            text = re.sub('http\S+\s*', ' ', text)  # remove URLs
            text = re.sub('RT|cc', ' ', text)  # remove RT and cc
            text = re.sub('@[^\s]+', ' ', text)
            text = clean_hashtag(text)
            text = clean_emoji(text)
            return text

        def clean_text(text):
            # Clean for tweets
            text = clean_tweet(text)
            # Remove punctuations
            text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)  # remove punctuation
            # remove extra whitespace
            text = re.sub('\s+', ' ', text)
            # Remove Emojis
            text = remove_emoji(text)
            # Convert text to lowercases
            text = text.lower()
            # Arabisy the text
            text = to_arabic(text)
            # Remove stop words
            text = remove_stop_words(text)
            # Remove numbers
            text = re.sub("\d+", " ", text)
            # Remove Tashkeel
            text = normalizeArabic(text)
            # text = re.sub('\W+', ' ', text)
            text = re.sub('[A-Za-z]+', ' ', text)
            text = re.sub(r'\\u[A-Za-z0-9\\]+', ' ', text)
            # remove extra whitespace
            text = re.sub('\s+', ' ', text)
            # Stemming
            # text = stem(text)
            return text

        df['tweet'] = df['tweet'].map(clean_text)
        return df

    def prepare_data(self, username, auth):
        data = self.preprocessing(username, auth)
        grouped = data.groupby("username")
        result = grouped.agg({
            "tweet": " ".join,
            "Word_Coun": "sum",
            "Char_Count": "sum",
            "Avg_Char_Per_Word": "mean",
            "Emoji_Count": "sum"
        })
        result.columns = ["Full_Text", "total_words", "total_char", "avg_chr_per_word", "total_emojis"]
        X_num = result.iloc[:, 1:5].values
        ss = MinMaxScaler()
        X_num = ss.fit_transform(X=X_num)
        vectorizer = CountVectorizer()
        X_text = vectorizer.fit_transform(result['Full_Text'])
        result = pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out())
        result['total_words'] = [x[0] for x in X_num]
        result["total_char"] = [x[1] for x in X_num]
        result["avg_chr_per_word"] = [x[2] for x in X_num]
        result["total_emojis"] = [x[3] for x in X_num]
        x = []
        for i in self.features:
            if i in result.columns:
                x.append(result[i][0])
            else:
                x.append(0)
        x = pd.DataFrame([x], columns=self.features)
        return x

    def predict(self, username, auth):
        def predict_batch(model, X):
            return model.predict_proba(X)[0][1]
        x = self.prepare_data(username, auth)
        models = [self.modelo, self.modelc, self.modele, self.modela, self.modeln]
        results = Parallel(n_jobs=-1)(delayed(predict_batch)(model, x) for model in models)
        #o = self.modelo.predict_proba(x)[0][1]
        #c = self.modelc.predict_proba(x)[0][1]
        #e = self.modele.predict_proba(x)[0][1]
        #a = self.modela.predict_proba(x)[0][1]
        #n = self.modeln.predict_proba(x)[0][1]
        #l = [o, c, e, a, n]
        #return l
        return results


def plot(x, y):
    plt.figure(facecolor='#162938')
    plt.pie(np.multiply(x, y),
            labels=['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'],
            explode=[0.05, 0, 0, 0, 0], shadow=True, autopct='%1.1f%%',
            textprops={'fontsize': 14, 'weight': 'bold', 'color': "w"})
    plt.savefig('static/plot.png')


t = {
    0: ['Openness',
        "'Openness' reflects a person's willingness to explore new ideas, emotions and experiences. People who "
        "score high in openness tend to be curious, imaginative, and creative. They may enjoy art, "
        "music and literature, and be more open to different cultures and ways of thinking."],
    1: ['Conscientiousness',
        "'Conscientiousness' refers to a person's level of organization, responsibility and dependability. People who "
        "score high in conscientiousness tend to be reliable, hardworking and goal-oriented. They may be more "
        "likely to follow rules and adhere to social norms."],
    2: ['Extroversion',
        "'Extroversion' reflects a person's level of sociability, assertiveness and energy. People who score high in "
        "extroversion tend to be outgoing, talkative and energetic. They may enjoy socializing and being the "
        "center of attention, and may be more likely to take risks and seek out excitement."],
    3: ['Agreeableness',
        "'Agreeableness' refers to a person's level of compassion, empathy and cooperation. People who score high in "
        "agreeableness tend to be kind, helpful and compassionate. They may be more concerned with others' "
        "well-being and be more likely to avoid conflict."],
    4: ['Neuroticism',
        "'Neuroticism' reflects a person's level of emotional instability and sensitivity. People who score high in "
        "neuroticism tend to be more prone to negative emotions such as anxiety, sadness, and anger. They may be "
        "more reactive to stress and more likely to experience mood swings."]
}
prob = [[1, 0.2, 0.2, 0.2, 0.2],[0.2, 1, 0.2, 0.2, 0.2],[0.2, 0.2, 1, 0.2, 0.2],[0.2, 0.2, 0.2, 1, 0.2],[0.2, 0.2, 0.2, 0.2, 1]]
db = SQLAlchemy()


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(250), unique=True)
    trait = db.Column(db.String(250))


login_manager = LoginManager()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)
