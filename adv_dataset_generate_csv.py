from robust_speech.adversarial.utils import find_closest_length_string
import os
path = "/home/raphael/dataspace/outputs/robust_speech/attacks/adversarial_librispeech/test-clean-200.csv"
audio = "/home/raphael/dataspace/outputs/robust_speech/attacks/adversarial_librispeech/adv-0.04"
new_path = "/home/raphael/dataspace/outputs/robust_speech/attacks/adversarial_librispeech/manifest.txt"

target_candidates = [
    "LET ME SEE HOW CAN I BEGIN",
    "NOW GO I CAN'T KEEP MY EYES OPEN",
    "SO YOU ARE NOT A GRAVE DIGGER THEN",
    "HE HAD HARDLY THE STRENGTH TO STAMMER",
    "WHAT CAN THIS MEAN SHE SAID TO HERSELF",
    "NOT YEARS FOR SHE'S ONLY FIVE AND TWENTY",
    "WHAT DOES NOT A MAN UNDERGO FOR THE SAKE OF A CURE",
    "IT IS EASY ENOUGH WITH THE CHILD YOU WILL CARRY HER OUT",
    "POOR LITTLE MAN SAID THE LADY YOU MISS YOUR MOTHER DON'T YOU",
    "AT LAST THE LITTLE LIEUTENANT COULD BEAR THE ANXIETY NO LONGER",
    "TAKE THE MEAT OF ONE LARGE CRAB SCRAPING OUT ALL OF THE FAT FROM THE SHELL",
    "TIS A STRANGE CHANGE AND I AM VERY SORRY FOR IT BUT I'LL SWEAR I KNOW NOT HOW TO HELP IT",
    "THE BOURGEOIS DID NOT CARE MUCH ABOUT BEING BURIED IN THE VAUGIRARD IT HINTED AT POVERTY PERE LACHAISE IF YOU PLEASE"]

with open(path, "r") as source:
    with open(new_path, "w") as target:
        for line in source:
            key = line.split(",")[0]
            print(key)
            if os.path.exists(os.path.join(audio, key+"_adv.wav")):
                og_transcript = line.split(",")[-1][:-1]
                adv_transcript = find_closest_length_string(
                    og_transcript, target_candidates)
                new_line = ",".join([key, og_transcript, adv_transcript])+"\n"
                target.write(new_line)
