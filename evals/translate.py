from languages import languages
from datasets_.mgsm import translate_mgsm
from datasets_.arc import translate_arc

if __name__ == "__main__":
    translate_mgsm(languages)
    translate_arc(languages)