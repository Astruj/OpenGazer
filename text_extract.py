# Need to run the file as root for Keyboard Listener on Mac OSX
import time
import datetime
import sys
import mss
import pytesseract
# from pynput.mouse import Listener
from pynput.keyboard import Listener, Key
import numpy
import re
import hashlib
import cv2
import os
from PIL import Image
from reorderf import reorderFlow
import re
from bs4 import BeautifulSoup

class Screen(object):
    frames = {}
    scroll_value = False
    new_frame = True
    # portions = []
    summary = []
    dimensions = []
    start_time = 0
    # print('Window Dimensions', dimensions)
    # browser_width = 768*2
    mw_panel = 0
    monitor = {}
    pixel_info = 0
    factor = 40
    n = 0
    data = {}  # text: coordinates, scroll
    dynamic_check = False

    def extract_Text_From_Image(self, image):
        text = pytesseract.image_to_string(image, lang='eng')
        return text

    def get_Coordinates(self):

        # return [[160, 140, 1439, 280],[0, 1160, 3000, 1160 + 250]]
        # (0, 150, 2500, 280)
        # return [[200, 140, 1700, 350],[2150,170,370+2150,730]]
        box = [[2150, 170, 370 + 2150, 730], [845, 830, 845 + 1700, 830 + 230],
               [0, 1160 + 50, 1700, 1160 + 180], [200, 140, 1700, 300]]
        # if self.dynamic_check:
        #     self.portions = [[200, 140, 1700, 300]]
        # else:
        #     self.portions = [[845, 830, 845+1700, 830+230]]
        #     self.dynamic_check = True
        return box

    def complete_The_Text(self, extracted_text, window, read_box):
        test_portion = read_box[:]
        test_portion[2] += 20
        if test_portion[0] >= 0:
            test_portion[0] -= 20
        x = self.extract_Text_From_Image(window.crop(test_portion))
        if len(extracted_text) == len(x):
            print('Did not find any extra text.')
            return extracted_text, read_box
        else:
            last_word = '.'
            read_box[0] = 0
            read_box[2] = window.width
            full_text = self.extract_Text_From_Image(window.crop(read_box))
            first_word_in_full_text = full_text.split(' ', 1)[0]
            try:
                if first_word_in_full_text.islower():
                    first_position = full_text.index('.') + 1
                else:
                    first_position = 0
            except ValueError:
                # print('why tho')
                first_position = 0
            try:
                last_position = full_text.rindex(last_word)
            except ValueError:
                last_position = len(full_text) + 1
            complete_text = full_text[first_position:last_position]
            return complete_text, read_box

    def text_Extraction(self, window_name, boxes):
        start = time.time()
        window = Image.open(window_name)
        for i in range(len(boxes)):
            read_box = boxes[i]
            extracted_image = window.crop(read_box)
            # extracted_image.show()
            extracted_text = self.extract_Text_From_Image(extracted_image)

            if read_box[2] - read_box[0] >= 2500:
                # extract.write(extracted_text+'\n\n')
                # print('\n' + extracted_text + '\n \n' + str(read_box))
                self.data[extracted_text] = (read_box, self.pixel_info)
            elif extracted_text:
                corrected_text, b_box = self.complete_The_Text(extracted_text, window, read_box)
                self.data[corrected_text] = (b_box, self.pixel_info)
                # extract.write(corrected_text+'\n\n')
                # print('\n' + extracted_text + '\n \n' + str(b_box))

        time_elapsed = time.time() - start
        print('Time taken:', time_elapsed)
        return

     # @staticmethod
    def clean_Extracted_Text(self, summary_unclean):

        #remove non-ascii characters
        non_ascii_removed = re.sub(r'[^\x00-\x7F]+', ' ', summary_unclean)
        #replace deciamal with underscore
        decmark_reg = re.compile('(?<=\d)\.(?=\d)')
        decimal_replaced = decmark_reg.sub('__', non_ascii_removed)
        #remove unwanted wild characters
        bad_chars = ['!', '*', '`', "'", '"', '[', ']', '(', ')', "?", "=", "~"]
        for i in bad_chars:
            wild_removed = ''.join(i for i in decimal_replaced if not i in bad_chars)
        wild_removed = re.sub('[!{[|]\S+[!\}\]lI1]?|\d+[\]lJ]+', '', wild_removed)
        # single letters removed
        singles_removed = ' '.join([w for w in wild_removed.split() if (len(w) > 1 and w not in ['a'])])
        #contract possible words
        contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                               "could've": "could have", "couldn't": "could not",

                               "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                               "hasn't": "has not", "haven't": "have not",

                               "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                               "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                               "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                               "I'm": "I am", "I've": "I have", "i'd": "i would",

                               "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                               "i've": "i have", "isn't": "is not", "it'd": "it would",

                               "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                               "it's": "it is", "let's": "let us", "ma'am": "madam",

                               "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                               "mightn't've": "might not have", "must've": "must have",

                               "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                               "needn't've": "need not have", "o'clock": "of the clock",

                               "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                               "sha'n't": "shall not", "shan't've": "shall not have",

                               "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                               "she'll've": "she will have", "she's": "she is",

                               "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                               "so've": "so have", "so's": "so as",

                               "this's": "this is", "that'd": "that would", "that'd've": "that would have",
                               "that's": "that is", "there'd": "there would",

                               "there'd've": "there would have", "there's": "there is", "here's": "here is",
                               "they'd": "they would", "they'd've": "they would have",

                               "they'll": "they will", "they'll've": "they will have", "they're": "they are",
                               "they've": "they have", "to've": "to have",

                               "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                               "we'll've": "we will have", "we're": "we are",

                               "we've": "we have", "weren't": "were not", "what'll": "what will",
                               "what'll've": "what will have", "what're": "what are",

                               "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
                               "where'd": "where did", "where's": "where is",

                               "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                               "who's": "who is", "who've": "who have",

                               "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                               "won't've": "will not have",

                               "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                               "y'all": "you all",

                               "y'all'd": "you all would", "y'all'd've": "you all would have",
                               "y'all're": "you all are", "y'all've": "you all have",

                               "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                               "you'll've": "you will have",

                               "you're": "you are", "you've": "you have"}
        contraction_done = ' '.join(
            [contraction_mapping[t] if t in contraction_mapping else t for t in singles_removed.split(" ")])
        #capitalize first character of sentence
        clean_summary = '. '.join(map(lambda s: s.strip().capitalize(), contraction_done.split('.')))

        print('_____________________CLEANING done___________________________________')
        return clean_summary

    @staticmethod
    def remove_Redundant_Text(sorted_portions):
        list_of_sentences = sorted_portions.split('\n')
        completed_lines_hash = set()
        redundancy_free_text = ''

        for line in list_of_sentences:
            words = line.split(" ")
            if len(words) < 5:
                continue
            hashValue = hashlib.md5(line.rstrip().encode('utf-8')).hexdigest()
            if hashValue not in completed_lines_hash:
                redundancy_free_text += line
                completed_lines_hash.add(hashValue)
        return redundancy_free_text

    # def get_wiki_text(self):
    #     import requests
    #     response = requests.get(
    #         'https://en.wikipedia.org/w/api.php',
    #         params={
    #             'action': 'query',
    #             'format': 'json',
    #             'titles': 'Bla Bla Bla',
    #             'prop': 'extracts',
    #             'exintro': True,
    #             'explaintext': True,
    #         }
    #     ).json()


if __name__ == "__main__":
    sc = Screen()

    for filename in os.listdir("Frames"):
        if filename.endswith(".png"):
            net = "Frames/" + filename
            frame = cv2.imread("Frames/" + filename)
            portions = sc.get_Coordinates()
            sc.text_Extraction(net, portions)
            continue
        else:
            continue
    # send data for reordering
    sorted_portions = reorderFlow(sc.data)
    # print('reordered text', sorted_portions)

    # remove redundancy here
    redundancy_free_text = sc.remove_Redundant_Text(sorted_portions)
    # print('\n redundancy free text: \n', redundancy_free_text)
    with open('extracted_summary1.txt', 'w+') as s:
        summary = sc.clean_Extracted_Text(redundancy_free_text)
        for each in summary.split('.'):
            each = each.split("__") # rectify floats
            each = ".".join(each)
            s.write(each + '.')
        s.close()
