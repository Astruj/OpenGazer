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
from PIL import Image
from reorderf import reorderFlow
import window_dimensions
import settings
class Screen(object):
    frames = {}
    scroll_value = False
    new_frame = True
    portions=[]
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
    def __init__(self):
        # self.get_Window_Dimensions()
        # self.dimensions = window_dimensions.get_active_window_dimensions()
        # self.monitor = {"top": self.dimensions[1], "left": self.dimensions[0]+self.mw_panel,
        #                 "width": self.dimensions[2], "height": self.dimensions[3]}
        self.get_Pixels_Moved_For_Each_Keystroke()

    def get_Pixels_Moved_For_Each_Keystroke(self):
        self.factor = 40
        return

    def get_Specific_Window(self):
        print(self.monitor)
        # frame = sct.grab(monitor)
        frame = numpy.array(sct.grab(self.monitor))
        # window = smp.toimage(frame)
        window = Image.fromarray(frame)
        window = window.convert('LA')
        # print('mss grabbed frame size: ',window.size)
        return window


    def extract_Text_From_Image(self, image):
        text = pytesseract.image_to_string(image, lang='eng')
        return text

    def get_Coordinates(self):

        # return [[160, 140, 1439, 280],[0, 1160, 3000, 1160 + 250]]
        # (0, 150, 2500, 280)
        # return [[200, 140, 1700, 350],[2150,170,370+2150,730]]
        self.portions = [[2150, 170, 370+2150, 730], [845, 830, 845+1700, 830+230],
                         [0, 1160+50, 1700, 1160+180], [200, 140, 1700, 300]]
        # if self.dynamic_check:
        #     self.portions = [[200, 140, 1700, 300]]
        # else:
        #     self.portions = [[845, 830, 845+1700, 830+230]]
        #     self.dynamic_check = True
        return


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
                    first_position = full_text.index('.')+1
                else:
                    first_position = 0
            except ValueError:
                # print('why tho')
                first_position = 0
            try:
                last_position = full_text.rindex(last_word)
            except ValueError:
                last_position = len(full_text)+1
            complete_text = full_text[first_position:last_position]
            return complete_text, read_box

    def text_Extraction(self, window, portions):
        if not self.scroll_value:
            # window.show()
            start = time.time()
            for i in range(len(portions)):
                read_box = portions[i]
                extracted_image = window.crop(read_box)
                # extracted_image.show()
                extracted_text = self.extract_Text_From_Image(extracted_image)
                if read_box[2]-read_box[0] >= 2500:
                    # extract.write(extracted_text+'\n\n')
                    print('\n'+extracted_text + '\n \n'+str(read_box))
                    self.data[extracted_text] = (read_box, self.pixel_info)
                elif extracted_text:
                    corrected_text, b_box = self.complete_The_Text(extracted_text, window, read_box)
                    self.data[corrected_text] = (b_box, self.pixel_info)
                    # extract.write(corrected_text+'\n\n')
                    print('\n'+extracted_text + '\n \n'+str(b_box))


            time_elapsed = time.time()-start
            print('Time taken:',time_elapsed)
            return
        else:
            return

    @staticmethod
    def clean_Extracted_Text(summary):
        clean_summary=''
        print('_____________________AFTER ClEANING___________________________________')
        # summary = re.sub('[\!\[\(]+\S*[\]\!]*|\S?\d*[\]\!\)\|]', '', summary)
        summary = re.sub('[!{[|]\S+[!\}\]lI1]?|\d+[\]lJ]+', '', summary)
        for each in summary:
            match = re.search("^(\w|\s|\'|\"|\.|\,|\-|\=|).*", each)
            if match:
                # print(each, end='')
                clean_summary += each
        return clean_summary

    def capture_Frames(self):
        threshold_start = time.time()
        while not self.scroll_value:
            last_time = time.time()
            threshold = int(last_time-threshold_start)

            if settings.keepRecording == False:
                break
            # Exit after getting First Frame
            # if n == 1:
            #     self.new_frame = False
                # break
            # if not self.scroll_value and threshold > 10:
            # if self.new_frame:
            #     window = self.get_Specific_Window()
            #     self.new_frame = False
                # The Frame Rate:
                # print("fps: {}".format(1 / (time.time() - last_time)))
            if threshold > 5 and self.new_frame:
                # n += 1
                self.n += 1
                window = self.get_Specific_Window()
                frame_name = 'frame_'+str(self.n)
                self.frames[frame_name] = [window, str(datetime.timedelta(seconds=int(time.time() - self.start_time)))]
                # window.show()
                self.new_frame = False
                # self.get_Coordinates()
                # self.text_Extraction(window, self.portions)
                break
            # else:
            #     # print('STAAAAAAAAHHHHHHP')
            #     break
        # self.scroll_value = False
        return

    @staticmethod
    def remove_Redundant_Text(sorted_portions):
        list_of_sentences = sorted_portions.split('\n')
        completed_lines_hash = set()
        redundancy_free_text = ''

        for line in list_of_sentences:
            hashValue = hashlib.md5(line.rstrip().encode('utf-8')).hexdigest()
            if hashValue not in completed_lines_hash:
                redundancy_free_text += line + ' '
                completed_lines_hash.add(hashValue)
        return redundancy_free_text

    def on_scroll(self, x, y, dx, dy):
        # print('scrolling')
        # print(x,y,dx,dy)

        self.scroll_value = True
        self.new_frame = True

    def on_press(self, key):
        if key == Key.down or key == Key.up:
            self.scroll_value = True
            self.new_frame = False
        if key == Key.down:
            self.pixel_info += self.factor
        elif key == Key.up and self.pixel_info - self.factor > 0:
            self.pixel_info -= self.factor
        # print(sc.new_frame)
        # print('{0} pressed'.format(
        #     key))

    def on_release(self, key):
        if key == Key.down or key == Key.up:
            self.new_frame = True
            self.scroll_value = False
        print('{0} release'.format(
        key))
        if key == Key.esc:
            # Stop listener
            return False

    def on_stop_recording(self):
        # send data for reordering
        sorted_portions = reorderFlow(self.data)
        print('reordered text', sorted_portions)

        # remove redundancy here
        redundancy_free_text = self.remove_Redundant_Text(sorted_portions)
        print('\n redundancyfree text: \n', redundancy_free_text)
        with open('extracted_summary.txt', 'w+') as s:
            summary = self.clean_Extracted_Text(redundancy_free_text)
            for each in summary.split('.'):
                s.write(each+'\n')
            s.close()

    def start_recording(self):
        listener = Listener(on_press=self.on_press, on_release=self.on_release)

        self.start_time = time.time()
        listener.start()


        self.dimensions = window_dimensions.get_active_window_dimensions()
        self.monitor = {"top": self.dimensions[1], "left": self.dimensions[0]+self.mw_panel,
                        "width": self.dimensions[2], "height": self.dimensions[3]}
        while True:
                if not settings.keepRecording:
                    print(settings.keepRecording)
                    break
                if self.new_frame:
                    print('Recording!')
                    self.capture_Frames()
                    # time.sleep(0.1)




if __name__ == "__main__":
    sc = Screen()
    with mss.mss() as sct:
        try:
            sc.start_recording()
        except KeyboardInterrupt:
            sc.on_stop_recording()
else:
    sct = mss.mss()
