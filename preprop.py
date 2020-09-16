import numpy as np
import pandas as pd
import os
import re


def phone_number_filter(text):
    re_pattern = r'\d{2,3}[-\.\s]*\d{3,4}[-\.\s]*\d{4}(?!\d)'
    text = re.sub(re_pattern, ' tel ', text)
    re_pattern = r'\(\d{3}\)\s*\d{4}[-\.\s]??\d{4}'
    text = re.sub(re_pattern, ' tel ', text)
    return text

def url_filter(text):
    re_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),|]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(re_pattern, 'url', text)
    return text
    
def price_filter(text):
    re_pattern = r'\d{1,3}[,\.]\d{1,3}[만\천]?\s?[원]|\d{1,5}[만\천]?\s?[원]'
    text = re.sub(re_pattern, ' money ', text)
    re_pattern = r'[일/이/삼/사/오/육/칠/팔/구/십/백][만\천]\s?[원]'
    text = re.sub(re_pattern, ' money ', text)
    re_pattern = r'(?!-)\d{2,4}[0]{2,4}(?!년)(?!.)|\d{1,3}[,/.]\d{3}'
    text = re.sub(re_pattern, ' money ', text)
    return text  

def date_filter(text):
    re_pattern = r'\d{1,2}[ 개]?[달\월]'
    text = re.sub(re_pattern, ' month ', text)
    re_pattern = r'\d{1}[ ]?[주]'
    text = re.sub(re_pattern, ' month ', text)
    re_pattern = r'\d{1}[ ]?[일]'
    text = re.sub(re_pattern, ' month ', text)
        
    re_pattern = r'\d{1}[ ]?[년]'
    text = re.sub(re_pattern, ' year ', text)
    re_pattern = r'[월화수목금토일][요][일]'
    text = re.sub(re_pattern, ' day ', text)
    re_pattern = r'[0-9]+[/][0-9]'
    text = re.sub(re_pattern, ' ', text)   
#     text = re.sub()
    return text   

def time_filter(text):
#     re_pattern = r'\d{1,2}[ ]?[시][간]'
#     text = re.sub(re_pattern, 'time', text)    
    re_pattern = r'\d{1,2}[ ]?[시간|시|분|hr|hour|minute|min]+'
    text = re.sub(re_pattern, ' time ', text)
    re_pattern = r'[0-9]+[:][0-9]+'
    text = re.sub(re_pattern, ' ', text)    
    return text   

def prescription_filter(text):
    re_pattern = '[0-9]+[ .]*[0-9]*[m|ml|mg|mEqN|timel|timeg]+[ \/]*[kg|mg|hr|h|g|lg]*[ \/]*[kg|mg|hr|h|g|lg]*'
    text = re.sub(re_pattern, ' 처방 ', text)
#     re_pattern = '.timel/kg'
#     text = re.sub(re_pattern, '', text)    
#     re_pattern = '.timel/kg'
#     text = re.sub(re_pattern, '', text)       
#     re_pattern = '[0-9]+[ .][0-9]+[m|ml|mg|mEqN]+[ \/][kg|mg|hr|h]+'
#     text = re.sub(re_pattern, '처방', text)    
    return text 

def remove_name(text) :
    re_pattern = '[\[].*[모니터][ ]*[b][y][ ]*[가-힣][ ]*[\]]'
    text = re.sub(re_pattern, ' ', text)
    re_pattern = '[야][간][ ]*[모][니][터][ ]*[b][y][ ]*[가-힣][가-힣][가-힣]'
    text = re.sub(re_pattern, ' ', text)
    re_pattern = '[ ]*[모][니][터][ ]*[b][y][ ]*[가-힣][가-힣][가-힣]'
    text = re.sub(re_pattern, '모니터', text)    
    return text

def preprocess(text) :
    text = phone_number_filter(text)
    text = url_filter(text)
    text = price_filter(text)
    text = date_filter(text)
    text = time_filter(text)
    text = remove_name(text)
#     text = prescription_filter(text)
    text = text.lower()
    text = re.sub('[ ][a-z][ ]','', text )
    text = re.sub('[ ][0-9]+[ ]','', text )
#     text = re.sub('[ ][0-9][.][ ]','', text )
    text = re.sub('[0-9]+[.][0-9]+','', text )
    text = re.sub('[0-9]+[.]','', text )
    text = re.sub('[.][0-9]+','', text )
    text = re.sub('[^A-Za-z가-힣.]',' ', text )
    text = re.sub('[\s]+[a-z][\s]+', ' ', text )
    text = re.sub('[.]+', '.', text )
    text = re.sub('[-]+', '-', text )
    text = re.sub('[ ]+', ' ', text )
    
    return text


def label_regex (x) :
    x = str(x)
    x = re.sub('중성화수술','중성화', x)
    x = re.sub('중성화 수술','중성화', x)
    x = re.sub('중성화','중성화', x)
    x = re.sub('여아중성화','중성화', x)
    x = re.sub('남아중성화','중성화', x)
    x = re.sub('여아Neutral','중성화', x)#여아Neutral
    x = re.sub('남아Neutral','중성화', x)
    x = re.sub('여중','중성화', x)
    x = re.sub('남중','중성화', x)

    x = re.sub('[건강][가-힣 ()a-zA-Z,0-9\/]+','건강검진', x)

    x = re.sub('X','normal', x)
    x = re.sub('x','normal', x)
    x = re.sub('이상없음','normal', x)
    x = re.sub('XX','normal', x)
    x = re.sub('-','normal', x)
    x = re.sub('normalnormal','normal', x)
    x = re.sub('정상','normal', x)
    x = re.sub('없음','normal', x)

    x = re.sub('스켈링','스케일링', x)

    x = re.sub('[.]0','', x)
    x = re.sub('[ ]+','', x)

    x = re.sub('[(].+[)]', '', x)
    return x