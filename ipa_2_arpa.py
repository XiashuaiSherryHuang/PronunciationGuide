'''This file is to covert transcriptions in IPA to arpabet'''
# https://github.com/chdzq/ARPAbetAndIPAConvertor
# !pip install arpabetandipaconvertor

from arpabetandipaconvertor.phoneticarphabet2arpabet import PhoneticAlphabet2ARPAbetConvertor
import re
import os
import argparse
import logging
logging.basicConfig(filename='ipa_2_arpa.log', level=logging.DEBUG)


IPA_MAP = {

    'ʰ': '',  # Silent aspiration (no ARPAbet equivalent)

    'ʂ': 'ʃ',  # Voiceless retroflex fricative

    'l': 'l',  # Voiced alveolar lateral approximant

    'ɭ': 'l',  # 'ɭ' Retroflex lateral approximant

    'ǂ': '',  # Click (no ARPAbet equivalent)

    'e': 'ɛ',  # Close-mid front unrounded vowel

    'ə': 'ə',  # Mid-central vowel (schwa)

    'ɺ': 'l',  # Retroflex lateral flap/approximant (approximated as 'l')

    '9': 'ʊ',  # Near-close near-back rounded vowel

    '̹': '',  # More rounded (no ARPAbet equivalent)

    '̪': '',  # Dental (no ARPAbet equivalent)

    'ɲ': 'n',  # 'ɲ',   # Palatal nasal

    'o': 'oʊ',  # Close-mid back rounded vowel

    '͎': '',  # Nasalized (no ARPAbet equivalent)

    'F': 'f',  # Voiceless labiodental fricative

    '̆': '',  # Extra-short (no ARPAbet equivalent)

    'ʍ': 'ʍ',  # Voiceless labialized velar approximant

    'β': 'v',  # Voiced bilabial fricative

    's': 's',  # Voiceless alveolar fricative

    '̚': '',  # No audible release (no ARPAbet equivalent)

    'ɡ': 'ɡ',  # Voiced velar stop

    'N': 'ŋ',  # Velar nasal

    '˺': '',  # Syllabic (no ARPAbet equivalent)

    'ɓ': 'b',  # Voiced bilabial implosive

    'ʤ': 'dʒ',  # Voiced postalveolar affricate

    'ɛ': 'ɛ',  # Open-mid front unrounded vowel

    ## 'ʀ': 'ʀ',   # Voiced uvular trill
    'ʀ': 'ɹ',

    'ʁ': 'ɹ',

    'ә': 'ə',  # Mid-central vowel (schwa)

    'ɗ': 'd',  # Voiced alveolar implosive

    'ɘ': 'ə',  # Close-mid central unrounded vowel (approximated as 'ə')

    'g': 'ɡ',  # Voiced velar stop

    '̑': '',  # Advanced tongue root (no ARPAbet equivalent)

    '1': '',  # Primary stress (no ARPAbet equivalent)

    'ʃ': 'ʃ',  # Voiceless postalveolar fricative

    'E': 'ɛ',  # Close-mid front unrounded vowel

    'ɚ': 'ɝ',  # R-colored schwa

    'ʏ': 'ʊ',  # Near-close near-front rounded vowel

    '∫': 'ʃ',  # Voiceless postalveolar fricative

    'ɖ': 'd',  # Voiced retroflex stop

    'ː': '',  # Long (no ARPAbet equivalent)

    'ŋ': 'ŋ',  # Velar nasal

    'ʲ': '',  # Palatalized (no ARPAbet equivalent)

    '2': '',  # Secondary stress (no ARPAbet equivalent)

    '\t': '',  # Tab (no ARPAbet equivalent)

    'ɯ': 'ɯ',  # Close back unrounded vowel

    'ˠ': '',  # Velarized (no ARPAbet equivalent)

    'ˈ': '',  # Primary stress (no ARPAbet equivalent)

    '̱': '',  # Creaky voice (no ARPAbet equivalent)

    'ʧ': 'tʃ',  # Voiceless postalveolar affricate

    'ʝ': 'j',  # Voiced palatal fricative

    'ت': '',  # Arabic letter (no ARPAbet equivalent)

    'z': 'z',  # Voiced alveolar fricative

    'ɑ': 'ɑ',  # Open back unrounded vowel

    'ɪ': 'ɪ',  # Near-close near-front unrounded vowel

    'ʎ': 'ʎ',  # Palatal lateral approximant

    '3': '',  # Tertiary stress (no ARPAbet equivalent)

    'M': '',  # Syllabic nasal (no ARPAbet equivalent)

    'v': 'v',  # Voiced labiodental fricative

    '6': '',  # Secondary stress (no ARPAbet equivalent)

    'L': '',  # Dark l (no ARPAbet equivalent)

    ' ': '',  # Space (no ARPAbet equivalent)

    '̀': '',  # Low tone (no ARPAbet equivalent)

    '٤': '',  # Arabic numeral (no ARPAbet equivalent)

    'D': 'd',  # Voiced alveolar stop

    '5': '',  # Secondary stress (no ARPAbet equivalent)

    '̤': '',  # Breathy voice (no ARPAbet equivalent)

    'ǀ': '',  # Dental click (no ARPAbet equivalent)

    ',': '',  # Comma (no ARPAbet equivalent)

    'ɽ': 'ɾ',  # Retroflex flap

    'ɹ': 'ɹ',  # Voiced alveolar approximant

    'ɣ': 'j',  ##'ɣ',   # Voiced velar fricative

    'ð': 'ð',  # Voiced dental fricative

    'ʷ': '',  # Labialized (no ARPAbet equivalent)

    'ɤ': 'ɝ',  ##'ɜ',   # Close-mid back unrounded vowel (approximated as 'ɜ')

    'ɨ': 'ɪ',  # Close central unrounded vowel (approximated as 'ɪ')

    'n': 'n',  # Voiced alveolar nasal

    'ӕ': 'æ',  # Near-open front unrounded vowel

    'H': '',  # Aspirated (no ARPAbet equivalent)

    '̰': '',  # Velopharyngeal friction (no ARPAbet equivalent)

    'P': 'p',  # Voiceless bilabial stop

    'Y': '',  # Central vowel (no ARPAbet equivalent)

    'ɔ': 'ɔ',  # Open-mid back rounded vowel

    'ɝ': 'ɝ',  # R-colored open-mid central unrounded vowel

    'ɳ': 'n',  ##'ɳ',   # Retroflex nasal

    'Q': '',  # Schwa with hook (no ARPAbet equivalent)

    'f': 'f',  # Voiceless labiodental fricative

    '̙': '',  # Raised (no ARPAbet equivalent)

    'χ': 'h',  # 'χ',   # Voiceless uvular fricative

    'ˤ': '',  # Pharyngealized (no ARPAbet equivalent)

    'ـ': '',  # Arabic letter (no ARPAbet equivalent)

    'ũ': '',  # Tilde (no ARPAbet equivalent)

    'ɒ': 'ɑ',  # Open back rounded vowel (approximated as 'ɑ')

    'ʈ': 't',  # Voiceless retroflex stop

    'œ': 'ɛ',  # Open-mid front rounded vowel (approximated as 'ɛ')

    '̠': '',  # Retracted tongue root (no ARPAbet equivalent)

    'ʕ': 't',  ##'ʕ',   # Voiced pharyngeal fricative

    'ă': 'æ',  # Near-open front unrounded vowel (approximated as 'æ')

    '͂': '',  # High tone (no ARPAbet equivalent)

    'ç': 'ç',  # Voiceless palatal fricative

    '˞': '',  # Rhoticity (no ARPAbet equivalent)

    '̬': '',  # Voiced (no ARPAbet equivalent)

    'B': 'b',  # Voiced bilabial stop

    'O': 'oʊ',  # Close-mid back rounded vowel

    'h': 'h',  # Voiceless glottal fricative

    'θ': 'θ',  # Voiceless dental fricative

    'j': 'j',  # Voiced palatal approximant

    'ʐ': 'r',  # 'ʐ',   # Voiced retroflex fricative

    '7': '',  # Tertiary stress (no ARPAbet equivalent)

    'ˡ': '',  # Velarized or pharyngealized (no ARPAbet equivalent)

    '̞': '',  # Lowered (no ARPAbet equivalent)

    'u': 'u',  # Close back rounded vowel

    'ᶴ': '',  # African palatal click (no ARPAbet equivalent)

    'G': 'ɡ',  # Voiced velar fricative

    'õ': 'oʊ',  # Close-mid back rounded vowel

    'ã': 'æ',  # Near-open front unrounded vowel (approximated as 'æ')

    'a': 'æ',  # Near-open front unrounded vowel (approximated as 'æ')

    'ɾ': 'ɾ',  # Alveolar tap or flap

    'm': 'm',  # Voiced bilabial nasal

    'ǝ': 'ə',  # Mid-central vowel (schwa)

    'ˀ': '',  # Glottal stop (no ARPAbet equivalent)

    'ɰ': 'w',  # Voiced velar approximant (approximated as 'w')

    '̌': '',  # Caron (no ARPAbet equivalent)

    '̩': '',  # Syllabic (no ARPAbet equivalent)

    'V': '',  # Central vowel (no ARPAbet equivalent)

    '-': '',  # Hyphen (no ARPAbet equivalent)

    '\uf1c8': '',  # Unknown character (no ARPAbet equivalent)

    '͆': '',  # Voiceless (no ARPAbet equivalent)

    'c': 's',  # Voiceless palatal fricative (approximated as 's')

    'ɮ': 'z',  # Voiced alveolar lateral fricative (approximated as 'z')

    'ɜ': 'ɝ',  # R-colored open-mid central unrounded vowel

    'ʒ': 'ʒ',  # Voiced postalveolar fricative

    '̃': '',  # Nasal (no ARPAbet equivalent)

    'ɐ': 'ʌ',  # Near-open central vowel (approximated as 'ʌ')

    '͉': '',  # More rounded (no ARPAbet equivalent)

    'I': 'ɪ',  # Near-close near-front unrounded vowel (approximated as 'ɪ')

    'R': 'ɹ',  # Voiced alveolar approximant

    'C': 'ʃ',  # Voiceless postalveolar fricative (approximated as 'ʃ')

    'ç': 'ʃ',

    ### 'ʁ': 'ʁ',   # Voiced uvular fricative

    'J': 'dʒ',  # Voiced postalveolar affricate

    'W': 'w',  # Voiced labio-velar approximant

    'ɯ': 'w',

    'ñ': 'nj',  # Palatal nasal approximant (approximated as 'nj')

    '8': '',  # Tertiary stress (no ARPAbet equivalent)

    'ĩ': '',  # Tilde (no ARPAbet equivalent)

    'ɕ': 'ʃ',  # Voiceless alveolo-palatal fricative (approximated as 'ʃ')

    '͡': '',  # Affricate (no ARPAbet equivalent)

    'ˑ': '',  # Half-long (no ARPAbet equivalent)

    '˳': '',  # Syllabic (no ARPAbet equivalent)

    'ẽ': '',  # Tilde (no ARPAbet equivalent)

    '̻': '',  # Strong (no ARPAbet equivalent)

    'i': 'i',  # Close front unrounded vowel

    'ɞ': 'ʌ',  # Open-mid central rounded vowel (approximated as 'ʌ')

    'ʟ': 'l',  # Velar lateral approximant

    'r': 'ɹ',  # Voiced alveolar approximant

    '4': '',  # Secondary stress (no ARPAbet equivalent)

    'p': 'p',  # Voiceless bilabial stop

    'U': 'u',  # Close back rounded vowel

    'ˢ': '',  # Voiceless (no ARPAbet equivalent)

    'k': 'k',  # Voiceless velar stop

    'y': 'j',  # Voiced palatal approximant (approximated as 'j')

    '̝': '',  # Raised (no ARPAbet equivalent)

    '̺': '',  # Apical (no ARPAbet equivalent)

    'ʔ': 'ʔ',  # Glottal stop

    'S': 's',  # Voiceless alveolar fricative (approximated as 's')

    'd': 'd',  # Voiced alveolar stop

    'T': 't',  # Voiceless alveolar stop

    'x': 'h',  ##'x'# Voiceless velar fricative

    'w': 'w',  # Voiced labio-velar approximant

    'æ': 'æ',  # Near-open front unrounded vowel

    'ɵ': 'ə',  # Mid-central vowel (schwa) (approximated as 'ə')

    'ɬ': 'l',  # Voiceless alveolar lateral fricative (approximated as 'l')

    'ʑ': 'ʒ',  # Voiced alveolo-palatal fricative (approximated as 'ʒ')

    'ᵊ': '',  # Reduced (no ARPAbet equivalent)

    'ⅼ': 'l',  # Lowercase letter 'l' (approximated as 'l')

    't': 't',  # Voiceless alveolar stop

    'K': 'k',  # Voiceless velar stop

    '̮': '',  # Non-syllabic (no ARPAbet equivalent)

    'ʉ': 'ʊ',  # Close central rounded vowel (approximated as 'ʊ')

    ':': '',  # Length mark (no ARPAbet equivalent)

    '\n': '',  # Newline (no ARPAbet equivalent)

    '̥': '',  # Voiceless (no ARPAbet equivalent)

    '0': '',  # No stress (no ARPAbet equivalent)

    'ʊ': 'ʊ',  # Near-close near-back rounded vowel

    'ɸ': 'f',  # Voiceless bilabial fricative (approximated as 'f')

    'ʋ': 'v',  # Voiced labiodental approximant (approximated as 'v')

    '̘': '',  # Advanced (no ARPAbet equivalent)

    'ø': 'oʊ',  # Close-mid front rounded vowel (approximated as 'oʊ')

    'ɱ': 'm',  # Labiodental nasal

    'ʌ': 'ʌ',  # Open-mid back unrounded vowel

    '\xa0': '',  # Non-breaking space (no ARPAbet equivalent)

    '̜': '',  # Retracted (no ARPAbet equivalent)

    'ɻ': 'ɹ',  # Retroflex approximant

    'ŭ': '',  # Breve (no ARPAbet equivalent)

    'A': 'æ',  # Near-open front unrounded vowel (approximated as 'æ')

    'Z': 'z',  # Voiced alveolar fricative

    'é': '',  # Acute accent (no ARPAbet equivalent)

    'b': 'b',  # Voiced bilabial stop

    'ɦ': 'h',  # Voiced glottal fricative

    '̟': '',  # Advanced (no ARPAbet equivalent)

    'ʎ': 'lj'

}


class Convertor():

    _ipa_convertor = PhoneticAlphabet2ARPAbetConvertor()

    @staticmethod
    def convert_file(current_file):
        with open(current_file, "r") as file:
            text = file.read()
            text = re.sub(r'\n', ' ', text) ##remove the line separator
            actual_string = Convertor.extract_string_between_brackets(text)

        return actual_string

    @staticmethod
    def extract_string_between_brackets(input_string):
        pattern = r'\[(.*?)\]'
        matches = re.findall(pattern, input_string)
        return matches[0]

    @staticmethod
    def process_raw_ipa(ipa_raw_string):
      ipa_raw_string_list = re.split('\s', ipa_raw_string)
      ipa_processe_string_list = []
      for sr in ipa_raw_string_list:
        if len(sr) == 0:
          continue
        scl = [IPA_MAP.get(c) for c in sr]
        sc = ''.join(scl)
        ipa_processe_string_list.append(sc)
      ipa_processed_string = ' '.join(ipa_processe_string_list)
      return ipa_processed_string

    def ipa_2_arpa(self, ipa_processed_string):
      ipa_string_list = re.split('\s', ipa_processed_string)
      out_put_list = []
      for s in ipa_string_list:
        if len(s) == 0:
          continue
        #print(s)
        f = self._ipa_convertor.convert(s)
        output = ''.join(re.split('\s', f))
        #print(output)
        out_put_list.append(output)
      out_str = ' '.join(out_put_list)
      return out_str

    @staticmethod
    def write_text_to_file(filename, text):
        try:
            with open(filename, 'w') as file:
                file.write(text)
        except IOError:
            logging.error(f'-------------An error occurred while writing to {filename}.')


def main(input_dir, output_dir):
    arpa_convertor = Convertor()
    for filename in os.listdir(input_dir):
        try:
            logging.info(filename)
            if filename.endswith('.txt'):
                ipa_file_path = os.path.join(input_dir, filename)
                ipa_raw_string = arpa_convertor.convert_file(ipa_file_path)
                ipa_processed_string = arpa_convertor.process_raw_ipa(ipa_raw_string)
                arpa_string = arpa_convertor.ipa_2_arpa(ipa_processed_string)
                arpa_convertor.write_text_to_file(os.path.join(output_dir, filename), arpa_string)
        except Exception as e:
            logging.error(f'An error occurred while writing to {filename} {str(e)}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='transcripts_directory', help='directory of input IPA files')
    parser.add_argument('--output', dest='arpa_directory', help='directory of output ARPABET files')
    args = parser.parse_args()

    logging.info('top 10 input files: {}' .format(os.listdir(args.transcripts_directory)[:10]))
    main(input_dir=args.transcripts_directory, output_dir=args.arpa_directory)
