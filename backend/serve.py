import os, io, json, re
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import cross_origin
import numpy as np
from PIL import Image

# TensorFlow import may be heavy — keep it lazy if you want, but we load once here.
import tensorflow as tf

# -----------------------------
# CONFIG
# -----------------------------
# Absolute paths to avoid local/Render path issues
APP_ROOT = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(APP_ROOT, "static")
MODEL_DIR = os.path.join(APP_ROOT, "dog_model")
LABELS_PATH = os.path.join(APP_ROOT, "labels.json")

IMG_SIZE = 224

# -----------------------------
# APP
# -----------------------------
app = Flask(__name__, static_folder=STATIC_DIR)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"],
)


# -----------------------------
# MODEL LOAD (load once)
# -----------------------------
print("Loading model from:", MODEL_DIR)
model = tf.saved_model.load(MODEL_DIR)
print("Available signatures:", list(model.signatures.keys()))
infer = model.signatures.get("serving_default") or list(model.signatures.values())[0]
print("Model loaded.")

# -----------------------------
# LABELS
# -----------------------------
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"labels.json not found at {LABELS_PATH}. Move frontend/labels.json -> backend/labels.json")

with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)
idx_to_class = {int(v): k for k, v in class_indices.items()}

breed_descriptions = {
    "affenpinscher": "Affenpinschers are small but fearless terriers known for their monkey-like expressions and lively personality.",
    "Afghan_hound": "The Afghan Hound is an elegant breed with long flowing hair, famous for its dignified and independent nature.",
    "African_hunting_dog": "African Hunting Dogs are highly social pack animals known for incredible stamina and unique multi-colored coats.",
    "Airedale": "Airedale Terriers are the 'King of Terriers,' smart and versatile with a strong working background.",
    "American_Staffordshire_terrier": "AmStaffs are muscular, confident, and affectionate companions with a courageous heart.",
    "Appenzeller": "This Swiss mountain dog is energetic, agile, and highly alert, ideal for farm and herding work.",
    "Aspin": "The Aspin (Asong Pinoy) is a resilient, loyal native Filipino dog known for intelligence and adaptability.",
    "Australian_terrier": "Australian Terriers are small, spirited dogs with a big personality and keen alertness.",
    "basenji": "The Basenji is the 'barkless dog,' known for its quiet nature, curly tail, and cat-like behavior.",
    "basset": "Basset Hounds are low-slung scent hounds famous for long ears, soulful eyes, and gentle temperament.",
    "beagle": "Beagles are friendly, curious, and excellent scent hounds loved worldwide for their playful personalities.",
    "Bedlington_terrier": "Bedlington Terriers are lamb-like in appearance but strong, fast, and spirited.",
    "Bernese_mountain_dog": "Bernese Mountain Dogs are large, gentle giants known for loyalty and calm temperament.",
    "black-and-tan_coonhound": "These coonhounds are strong scent trackers built for endurance and night hunting.",
    "Blenheim_spaniel": "Blenheim Spaniels are affectionate toy dogs known for their soft coats and friendly temperament.",
    "bloodhound": "Bloodhounds are master trackers with unmatched scenting ability and gentle personalities.",
    "bluetick": "Bluetick Coonhounds are vigorous, vocal hunting dogs known for stamina and determination.",
    "Border_collie": "Border Collies are one of the most intelligent breeds—high-energy and exceptional at herding.",
    "Border_terrier": "Border Terriers are hardy and affectionate working dogs with strong prey drive and friendly nature.",
    "borzoi": "Borzois are graceful sighthounds known for their speed, elegance, and calm demeanor.",
    "Boston_bull": "Boston Terriers are lively, friendly, and compact dogs nicknamed the 'American Gentleman.'",
    "Bouvier_des_Flandres": "A powerful and intelligent working dog, known for its strong build and protective nature.",
    "boxer": "Boxers are playful, strong, and loyal dogs with boundless energy and affection for families.",
    "Brabancon_griffon": "These small, expressive dogs are loved for their human-like faces and affectionate behavior.",
    "briard": "Briards are intelligent French herding dogs with long coats and excellent protective instincts.",
    "Brittany_spaniel": "Brittanys are active, agile gun dogs prized for versatility and trainability.",
    "bull_mastiff": "Bullmastiffs are powerful yet gentle guardians known for loyalty and quiet confidence.",
    "cairn": "Cairn Terriers are small, curious, and brave dogs originally bred for hunting burrowing animals.",
    "Cardigan": "Cardigan Welsh Corgis are sturdy herding dogs with long bodies, big ears, and loyal personalities.",
    "Chesapeake_Bay_retriever": "Chessies are strong swimmers and intelligent retrievers with excellent outdoor stamina.",
    "Chihuahua": "Chihuahuas are tiny but bold, known for their lively and affectionate temperament.",
    "chow": "Chow Chows are known for their lion-like mane, blue tongue, and independent personality.",
    "clumber": "Clumber Spaniels are calm, gentle hunting dogs with heavy bones and affectionate nature.",
    "cocker_spaniel": "Cocker Spaniels are friendly, affectionate, and known for their beautiful long ears.",
    "collie": "Collies are gentle herding dogs known for loyalty, intelligence, and iconic long coats.",
    "curly-coated_retriever": "A strong, confident retriever with water-resistant curly fur and excellent working skills.",
    "Dandie_Dinmont": "A rare terrier breed known for long bodies, silky top knots, and affectionate personality.",
    "dhole": "Dholes are wild Asian dogs known for social pack behavior and strong hunting instincts.",
    "dingo": "Dingoes are wild, intelligent Australian canines known for agility and survival skills.",
    "Doberman": "Dobermans are loyal, athletic guard dogs known for intelligence and strong protective instincts.",
    "English_foxhound": "English Foxhounds are pack-oriented scent hounds bred for stamina and teamwork.",
    "English_setter": "Graceful, friendly sporting dogs known for endurance and beautiful feathered coats.",
    "English_springer": "Springer Spaniels are energetic hunting dogs known for intelligence and friendly nature.",
    "EntleBucher": "Entlebuchers are compact Swiss herd dogs with high energy and strong working drive.",
    "Eskimo_dog": "American Eskimo Dogs are fluffy, intelligent, and cheerful companions with bright white coats.",
    "flat-coated_retriever": "These retrievers are cheerful, energetic, and enthusiastic working dogs.",
    "French_bulldog": "French Bulldogs are loving, playful companion dogs adored for their bat ears and cute faces.",
    "German_shepherd": "German Shepherds are loyal, highly intelligent working dogs known for versatility and strength.",
    "German_short-haired_pointer": "An athletic hunting breed with excellent tracking skills and endless stamina.",
    "giant_schnauzer": "Giant Schnauzers are strong, loyal working dogs with protective instincts and high energy.",
    "golden_retriever": "Golden Retrievers are friendly, intelligent, and one of the most beloved family dogs in the world.",
    "Gordon_setter": "Gordon Setters are elegant, strong hunting dogs known for loyalty and endurance.",
    "Great_Dane": "Great Danes are gentle giants famous for their huge size and affectionate personalities.",
    "Great_Pyrenees": "Large, calm guardian dogs bred to protect livestock with patience and strength.",
    "Greater_Swiss_Mountain_dog": "Swiss mountain dogs known for strength, loyalty, and striking tri-color coats.",
    "groenendael": "Belgian Sheepdogs with strong working ability and elegant black coats.",
    "Ibizan_hound": "Ibizans are slim, fast hounds known for agility and strong prey drive.",
    "Irish_setter": "Irish Setters are friendly, energetic sporting dogs with striking red coats.",
    "Irish_terrier": "Courageous, lively terriers known for their fiery red coats and bold personality.",
    "Irish_water_spaniel": "Curly, water-loving spaniels known for intelligence and versatility.",
    "Irish_wolfhound": "These gentle giants are among the tallest dog breeds, known for calm and loyal temperament.",
    "Italian_greyhound": "Small, elegant sighthounds known for grace, speed, and affectionate nature.",
    "Japanese_spaniel": "A charming toy breed with expressive eyes and elegant behavior.",
    "keeshond": "Keeshonds are fluffy, affectionate dogs known for fox-like faces and friendly personality.",
    "kelpie": "Australian Kelpies are intelligent herding dogs with exceptional stamina.",
    "Kerry_blue_terrier": "A lively, strong terrier with unique blue coat and playful personality.",
    "komondor": "Famous for its corded coat, the Komondor is a strong guard dog built for protection.",
    "kuvasz": "Large, strong Hungarian guardians known for loyalty and courage.",
    "Labrador_retriever": "Labs are friendly, outgoing, and highly trainable—one of the world's most popular breeds.",
    "Lakeland_terrier": "A small, confident terrier known for courage and curiosity.",
    "Leonberg": "Huge, friendly gentle giants with lion-like appearance and calm personalities.",
    "Lhasa": "Lhasa Apsos are alert, confident companion dogs once used as Tibetan monastery guardians.",
    "malamute": "Alaskan Malamutes are powerful sled dogs known for endurance and strong build.",
    "malinois": "Belgian Malinois are elite working dogs known for speed, intelligence, and discipline.",
    "Maltese_dog": "Maltese are gentle, affectionate toy dogs known for their long white coats.",
    "Mexican_hairless": "Xolo dogs are ancient, hairless companions known for loyalty and warmth.",
    "miniature_pinscher": "A tiny but fearless breed known for energetic and confident behavior.",
    "miniature_poodle": "Mini Poodles are smart, elegant dogs known for their intelligence and hypoallergenic coats.",
    "miniature_schnauzer": "A spirited and friendly terrier with a distinctive beard and bushy eyebrows.",
    "Newfoundland": "Massive, sweet-natured water dogs known for strength and gentle temperament.",
    "Norfolk_terrier": "Small, sturdy terriers with lively and affectionate personalities.",
    "Norwegian_elk hound": "A hardy Nordic breed known for hunting ability and loyalty.",
    "Norwich_terrier": "Small, charming terriers known for playful and brave character.",
    "Old_English_sheepdog": "Large herding dogs with shaggy coats and friendly nature.",
    "otterhound": "A rare scent hound known for its shaggy coat and love for water.",
    "papillon": "Small, elegant toy dogs known for butterfly-like ears and playful nature.",
    "Pekinese": "A royal toy breed with lion-like mane and confident personality.",
    "Pembroke": "Pembroke Welsh Corgis are loyal herding dogs famous for short legs and big smiles.",
    "Philippine_Forest_dog": "A rare Filipino native dog known for agility, adaptability, and survival instincts.",
    "Pomeranian": "Small, fluffy dogs with big personalities and fox-like expressions.",
    "pug": "Pugs are charming, affectionate, and playful dogs with iconic wrinkly faces.",
    "redbone": "Redbone Coonhounds are strong hunting dogs known for stamina and sleek red coats.",
    "Rhodesian_ridgeback": "Famous for the ridge on their back, these dogs are strong, confident, and loyal.",
    "Rottweiler": "Rottweilers are powerful, intelligent working dogs known for loyalty and protective nature.",
    "Saint_Bernard": "Massive rescue dogs known for gentleness and iconic Swiss mountain heritage.",
    "Saluki": "One of the oldest breeds, Salukis are elegant sighthounds with incredible speed.",
    "Samoyed": "Samoyeds are fluffy white dogs with a friendly 'smiling' expression.",
    "schipperke": "Small, fox-like Belgian dogs known for agility and alertness.",
    "Scotch_terrier": "Bold terriers with distinctive beards and confident personalities.",
    "Scottish_deerhound": "Tall, elegant hounds bred for speed and gentle temperament.",
    "Sealyham_terrier": "Small terriers with strong personalities and charming expressions.",
    "Shetland_sheepdog": "Shelties are intelligent herding dogs with agility and loyalty.",
    "Shih-Tzu": "Shih Tzus are affectionate toy dogs known for long flowing coats and friendly nature.",
    "Siberian_husky": "Huskies are energetic, vocal sled dogs known for striking blue eyes and stamina.",
    "silky_terrier": "A small Australian terrier known for its silky coat and lively temperament.",
    "soft-coated_wheaten_terrier": "Friendly, spirited terriers with silky wheaten-colored coats.",
    "Staffordshire_bullterrier": "Compact, muscular terriers known for affectionate and brave temperament.",
    "standard_poodle": "Standard Poodles are highly intelligent, elegant dogs with hypoallergenic coats.",
    "standard_schnauzer": "Strong, medium-sized working dogs known for intelligence and versatility.",
    "Sussex_spaniel": "Calm, affectionate spaniels with golden liver-colored coats.",
    "Tibetan_mastiff": "Huge guardian dogs known for strength, independence, and loyalty.",
    "Tibetan_terrier": "A shaggy-coated breed known for sensitivity, playfulness, and companionship.",
    "toy_poodle": "Toy Poodles are tiny, intelligent, and graceful companions.",
    "toy_terrier": "Tiny terriers known for energy, alertness, and playful personality.",
    "vizsla": "Hungarian Vizslas are athletic, affectionate, and skilled hunting dogs.",
    "Walker_hound": "Strong tracking dogs known for speed, stamina, and loud baying.",
    "Weimaraner": "Sleek, silver-gray hunting dogs known for intelligence and athleticism.",
    "Welsh_springer_spaniel": "Friendly, energetic sporting dogs with distinctive red and white coats.",
    "West_Highland_white_terrier": "Westies are confident, friendly terriers with iconic white coats.",
    "whippet": "Slim, fast sighthounds known for gentle temperament and remarkable speed.",
    "wire-haired_fox_terrier": "Energetic terriers with wiry coats and spirited personality.",
    "Yorkshire_terrier": "Yorkies are tiny, confident dogs known for long silky coats and lively behavior."
}
breed_temperaments = {
    "affenpinscher": "Confident, bold, mischievous, and lively with a big-dog attitude.",
    "Afghan_hound": "Independent, aloof, gentle, and elegant with a regal personality.",
    "African_hunting_dog": "Highly social, cooperative, energetic, and relentless hunters.",
    "Airedale": "Intelligent, confident, energetic, and versatile with strong drive.",
    "American_Staffordshire_terrier": "Courageous, loyal, affectionate, and people-oriented.",
    "Appenzeller": "Energetic, alert, hardworking, and highly active.",
    "Aspin": "Loyal, smart, adaptable, resilient, and friendly with humans.",
    "Australian_terrier": "Alert, spirited, affectionate, and courageous for its size.",
    "basenji": "Independent, quiet, intelligent, and cat-like in behavior.",
    "basset": "Gentle, patient, sweet-tempered, and determined scent hound.",
    "beagle": "Friendly, curious, playful, and food-driven with strong scent instinct.",
    "Bedlington_terrier": "Affectionate, spirited, alert, and surprisingly athletic.",
    "Bernese_mountain_dog": "Gentle, calm, loyal, and excellent with families.",
    "black-and-tan_coonhound": "Outgoing, friendly, determined, and strong-willed tracker.",
    "Blenheim_spaniel": "Affectionate, gentle, friendly, and eager to please.",
    "bloodhound": "Gentle, patient, determined, and strong-nosed but stubborn.",
    "bluetick": "Alert, vocal, determined, and energetic with strong hunting drive.",
    "Border_collie": "Extremely intelligent, energetic, focused, and highly trainable.",
    "Border_terrier": "Hardy, affectionate, plucky, and friendly with people.",
    "borzoi": "Calm, gentle, independent, and dignified.",
    "Boston_bull": "Friendly, lively, smart, and people-oriented.",
    "Bouvier_des_Flandres": "Protective, loyal, calm, and confident working dog.",
    "boxer": "Playful, energetic, affectionate, and loyal to family.",
    "Brabancon_griffon": "Expressive, affectionate, sensitive, and people-centered.",
    "briard": "Loyal, intelligent, protective, and strong-willed herder.",
    "Brittany_spaniel": "Energetic, friendly, eager to please, and highly trainable.",
    "bull_mastiff": "Calm, loyal, protective, and gentle with family.",
    "cairn": "Curious, alert, brave, and full of terrier personality.",
    "Cardigan": "Loyal, intelligent, stable, and hardworking herding dog.",
    "Chesapeake_Bay_retriever": "Independent, strong-willed, loyal, and excellent working dog.",
    "Chihuahua": "Bold, lively, loyal, and affectionate with owners.",
    "chow": "Independent, reserved, loyal, and dignified.",
    "clumber": "Gentle, calm, steady, and affectionate family dog.",
    "cocker_spaniel": "Friendly, sweet, eager to please, and gentle.",
    "collie": "Loyal, intelligent, gentle, and excellent with families.",
    "curly-coated_retriever": "Confident, calm, intelligent, and independent.",
    "Dandie_Dinmont": "Gentle, affectionate, bold, and independent terrier.",
    "dhole": "Highly social, intelligent, cooperative, and energetic hunter.",
    "dingo": "Alert, wary, intelligent, and independent.",
    "Doberman": "Loyal, fearless, intelligent, and highly trainable.",
    "English_foxhound": "Friendly, sociable, active, and pack-driven.",
    "English_setter": "Gentle, friendly, energetic, and affectionate.",
    "English_springer": "Energetic, friendly, eager to please, and smart.",
    "EntleBucher": "Energetic, alert, confident, and hardworking.",
    "Eskimo_dog": "Friendly, smart, energetic, and cheerful.",
    "flat-coated_retriever": "Happy, outgoing, enthusiastic, and playful.",
    "French_bulldog": "Playful, affectionate, sociable, and calm indoors.",
    "German_shepherd": "Intelligent, loyal, confident, and highly trainable.",
    "German_short-haired_pointer": "Energetic, eager, smart, and versatile working dog.",
    "giant_schnauzer": "Energetic, loyal, protective, and powerful worker.",
    "golden_retriever": "Gentle, friendly, smart, and eager to please.",
    "Gordon_setter": "Loyal, alert, intelligent, and steady hunter.",
    "Great_Dane": "Gentle, friendly, patient, and affectionate giant.",
    "Great_Pyrenees": "Calm, patient, protective, and independent guardian.",
    "Greater_Swiss_Mountain_dog": "Loyal, alert, strong, and steady family guardian.",
    "groenendael": "Intelligent, energetic, alert, and loyal.",
    "Ibizan_hound": "Agile, gentle, playful, and independent-minded.",
    "Irish_setter": "Friendly, energetic, outgoing, and loving.",
    "Irish_terrier": "Bold, spirited, loyal, and fiery but affectionate.",
    "Irish_water_spaniel": "Intelligent, energetic, playful, and versatile.",
    "Irish_wolfhound": "Gentle, calm, patient, and affectionate giant.",
    "Italian_greyhound": "Sensitive, affectionate, playful, and gentle.",
    "Japanese_spaniel": "Elegant, affectionate, alert, and cat-like.",
    "keeshond": "Friendly, bright, alert, and very people-oriented.",
    "kelpie": "Energetic, intelligent, hardworking, and driven.",
    "Kerry_blue_terrier": "Alert, lively, loyal, and strong-willed.",
    "komondor": "Protective, independent, calm, and strong guardian.",
    "kuvasz": "Loyal, protective, independent, and fearless.",
    "Labrador_retriever": "Friendly, playful, outgoing, and trainable.",
    "Lakeland_terrier": "Confident, bold, cheerful, and curious.",
    "Leonberg": "Gentle, calm, friendly, and very affectionate.",
    "Lhasa": "Alert, confident, independent, and loyal.",
    "malamute": "Friendly, strong-willed, energetic, and hardworking.",
    "malinois": "Extremely intelligent, intense, driven, and highly trainable.",
    "Maltese_dog": "Gentle, affectionate, playful, and friendly.",
    "Mexican_hairless": "Calm, loyal, alert, and affectionate.",
    "miniature_pinscher": "Fearless, energetic, confident, and lively.",
    "miniature_poodle": "Smart, lively, playful, and people-oriented.",
    "miniature_schnauzer": "Alert, friendly, energetic, and intelligent.",
    "Newfoundland": "Gentle, calm, patient, and extremely sweet-natured.",
    "Norfolk_terrier": "Friendly, spirited, loving, and sociable.",
    "Norwegian_elk hound": "Bold, loyal, alert, and energetic.",
    "Norwich_terrier": "Cheerful, brave, affectionate, and energetic.",
    "Old_English_sheepdog": "Gentle, adaptable, affectionate, and playful.",
    "otterhound": "Friendly, boisterous, independent, and scent-driven.",
    "papillon": "Happy, smart, lively, and trainable.",
    "Pekinese": "Independent, loyal, regal, and affectionate to owners.",
    "Pembroke": "Friendly, smart, alert, and hardworking.",
    "Philippine_Forest_dog": "Alert, agile, loyal, and adaptable to environments.",
    "Pomeranian": "Bold, lively, alert, and affectionate.",
    "pug": "Playful, charming, sociable, and affectionate.",
    "redbone": "Friendly, energetic, determined, and people-oriented.",
    "Rhodesian_ridgeback": "Independent, strong-willed, loyal, and confident.",
    "Rottweiler": "Loyal, confident, protective, and intelligent.",
    "Saint_Bernard": "Gentle, calm, patient, and affectionate.",
    "Saluki": "Reserved, gentle, independent, and graceful.",
    "Samoyed": "Friendly, alert, gentle, and people-loving.",
    "schipperke": "Alert, curious, bold, and energetic.",
    "Scotch_terrier": "Confident, independent, brave, and loyal.",
    "Scottish_deerhound": "Gentle, calm, dignified, and easygoing.",
    "Sealyham_terrier": "Alert, spirited, charming, and independent.",
    "Shetland_sheepdog": "Intelligent, loyal, alert, and gentle.",
    "Shih-Tzu": "Affectionate, friendly, calm, and people-loving.",
    "Siberian_husky": "Energetic, vocal, friendly, and independent.",
    "silky_terrier": "Alert, lively, intelligent, and friendly.",
    "soft-coated_wheaten_terrier": "Friendly, spirited, affectionate, and playful.",
    "Staffordshire_bullterrier": "Friendly, brave, affectionate, and people-loving.",
    "standard_poodle": "Very intelligent, elegant, energetic, and trainable.",
    "standard_schnauzer": "Bold, smart, energetic, and protective.",
    "Sussex_spaniel": "Gentle, calm, affectionate, and steady.",
    "Tibetan_mastiff": "Independent, protective, calm, and strong-willed.",
    "Tibetan_terrier": "Friendly, sensitive, playful, and loyal.",
    "toy_poodle": "Smart, lively, affectionate, and trainable.",
    "toy_terrier": "Energetic, alert, playful, and bold.",
    "vizsla": "Affectionate, loyal, energetic, and very people-oriented.",
    "Walker_hound": "Energetic, confident, vocal, and determined tracker.",
    "Weimaraner": "Energetic, intelligent, loyal, and strong-willed.",
    "Welsh_springer_spaniel": "Friendly, active, affectionate, and devoted.",
    "West_Highland_white_terrier": "Confident, friendly, bold, and spirited.",
    "whippet": "Gentle, affectionate, calm indoors, and playful.",
    "wire-haired_fox_terrier": "Energetic, bold, lively, and adventurous.",
    "Yorkshire_terrier": "Confident, lively, bold, and affectionate."
}
breed_appearance = {
    "affenpinscher": "Affenpinschers are small, sturdy terriers with a rough, wiry coat and a distinct monkey-like facial expression. They typically have a short muzzle, round dark eyes, and a compact body with a confident stance.",
    "Afghan_hound": "Afghan Hounds are tall, elegant sighthounds with very long, silky hair that drapes across their body. They have a refined head, high hipbones, and a sweeping tail curve that adds to their regal appearance.",
    "African_hunting_dog": "African Hunting Dogs have slender, muscular bodies with long legs built for endurance. Their coats are short and patterned with unique patches of black, tan, and white, giving each dog a one-of-a-kind appearance.",
    "Airedale": "Airedales are the largest of the terrier breeds, with a strong square frame and dense wiry coat. They have a long head, expressive eyes, and a distinctive beard that adds character.",
    "American_Staffordshire_terrier": "AmStaffs are medium-sized, muscular dogs with a broad chest and compact build. Their short coat is smooth and glossy, highlighting their athletic outline.",
    "Appenzeller": "Appenzellers are medium-sized Swiss working dogs with a sturdy frame and short, tricolor coat. They have a well-defined head, curled tail, and an alert, energetic expression.",
    "Aspin": "Aspin dogs vary in appearance but are generally medium-sized with a lean, balanced body and short coat suited for tropical climates. They often have pointed ears, deep chests, and agile structures.",
    "Australian_terrier": "Australian Terriers have a small, sturdy frame with a longish rough coat and distinctive ruff around the neck. Their erect ears and keen expression give them an alert look.",
    "basenji": "Basenjis are small, fine-boned dogs with a curled tail and short, sleek coat. Their wrinkled forehead and upright ears contribute to their unique, fox-like appearance.",
    "basset": "Basset Hounds have long, heavy bodies with very short legs and loose, wrinkled skin. Their long ears and soulful eyes create a gentle, expressive look.",
    "beagle": "Beagles are compact, muscular hounds with short coats and long, drooping ears. Their bright eyes and tricolor or bicolor patterns give them a cheerful appearance.",
    "Bedlington_terrier": "Bedlington Terriers have a distinctive lamb-like appearance with a curly, woolly coat and arched back. Their narrow head and topknot add to their unique silhouette.",
    "Bernese_mountain_dog": "Bernese Mountain Dogs are large, strong working dogs with a thick tricolor double coat. Their broad head, deep chest, and friendly expression make them visually striking.",
    "black-and-tan_coonhound": "Black-and-Tan Coonhounds are tall, athletic hounds with a sleek black coat accented by tan markings. Their long ears and deep chest emphasize their tracking ability.",
    "Blenheim_spaniel": "Blenheim Spaniels are small toy dogs with silky coats and feathered ears. They are known for their distinctive red-and-white coloration and gentle, expressive faces.",
    "bloodhound": "Bloodhounds are large scent hounds with loose, wrinkled skin and long drooping ears. Their deep-set eyes and powerful skeletal frame give them a noble yet solemn look.",
    "bluetick": "Bluetick Coonhounds have a muscular frame and a unique speckled blue coat pattern. Their long ears and athletic build reflect their purpose as endurance hunters.",
    "Border_collie": "Border Collies have a lean, agile build with a medium-length double coat that can be smooth or rough. Their intense eyes and alert posture enhance their working-dog appearance.",
    "Border_terrier": "Border Terriers are small, sturdy dogs with a wiry coat and otter-shaped head. Their narrow frame and slightly rough appearance suit their working origins.",
    "borzoi": "Borzoi are tall, graceful sighthounds with narrow heads and long, silky coats. Their deep chest and arched back give them an elegant, flowing silhouette.",
    "Boston_bull": "Boston Terriers are compact, tuxedo-patterned dogs with large round eyes and short muzzles. Their erect ears and square stance contribute to their lively, charming look.",
    "Bouvier_des_Flandres": "Bouviers are large, powerfully built working dogs with a thick, tousled coat. Their broad head, strong neck, and distinctive beard give them a rugged, imposing appearance.",
    "boxer": "Boxers are medium-to-large dogs with a muscular build and a strong, square jaw. Their short coat shows brindle or fawn patterns, and their expressive face radiates energy and alertness.",
    "Brabancon_griffon": "Brabancon Griffons are small toy dogs with a distinctive pushed-in face and expressive eyes. Their rough or smooth coat adds character to their compact, sturdy frame.",
    "briard": "Briards are large, long-haired herding dogs with a flowing, wavy coat. Their strong, agile frame and expressive eyes convey intelligence and attentiveness.",
    "Brittany_spaniel": "Brittany Spaniels are medium-sized sporting dogs with a dense, flat or wavy coat. Their expressive eyes, floppy ears, and balanced body reflect their energetic and agile nature.",
    "bull_mastiff": "Bullmastiffs are large, solidly built dogs with a broad head and muscular body. Their short coat and deep chest highlight their strength and imposing presence.",
    "cairn": "Cairn Terriers are small, compact dogs with a shaggy, weather-resistant coat. Their dark eyes and alert expression give them a lively and curious look.",
    "Cardigan": "Cardigan Welsh Corgis have long bodies, short legs, and a bushy tail. Their dense coat and large, expressive ears create a distinct, endearing appearance.",
    "Chesapeake_Bay_retriever": "Chessies are medium-to-large retrievers with a dense, wavy coat that repels water. Their muscular frame and amber eyes convey intelligence and stamina.",
    "Chihuahua": "Chihuahuas are tiny dogs with large, expressive eyes and erect ears. Their coat may be smooth or long, accentuating their delicate yet confident posture.",
    "chow": "Chow Chows are medium-to-large dogs with a thick mane-like coat and a broad skull. Their deep-set eyes and blue-black tongue make them instantly recognizable.",
    "clumber": "Clumber Spaniels are large, heavy-set dogs with a dense, straight coat. Their broad head and drooping lips give them a gentle, dignified expression.",
    "cocker_spaniel": "Cocker Spaniels are medium-sized dogs with long, silky ears and a soft, feathered coat. Their rounded eyes and compact frame give them a friendly, approachable appearance.",
    "collie": "Collies are medium-to-large herding dogs with a long, flowing coat and an elegant frame. Their intelligent eyes and wedge-shaped head convey alertness and loyalty.",
    "curly-coated_retriever": "Curly-Coated Retrievers are large dogs with a distinctive tight, curly coat. Their broad chest and strong legs emphasize their stamina and athletic build.",
    "Dandie_Dinmont": "Dandie Dinmont Terriers are small, elongated dogs with a soft, topknot-like coat on the head. Their low-slung frame and expressive eyes give them a unique charm.",
    "dhole": "Dholes are medium-sized wild dogs with lean, muscular bodies and short reddish coats. Their bushy tails and alert stance reflect agility and social hunting behavior.",
    "dingo": "Dingoes are medium-sized wild dogs with a lean frame and short, sandy-colored coat. Their erect ears, long muzzle, and bushy tail give them a distinctive wild appearance.",
    "Doberman": "Dobermans are medium-to-large dogs with a sleek, muscular build. Their short, smooth coat, wedge-shaped head, and alert ears convey strength and elegance.",
    "English_foxhound": "English Foxhounds are tall, athletic hounds with a short, dense coat. Their long ears, strong legs, and deep chest make them excellent for endurance hunting.",
    "English_setter": "English Setters are medium-to-large sporting dogs with a long, silky coat often speckled with markings. Their gentle eyes and graceful posture highlight elegance and agility.",
    "English_springer": "English Springer Spaniels are medium-sized sporting dogs with a strong, compact frame. Their coat is medium-length, often wavy, with feathering on legs, chest, and ears. Expressive eyes and alert ears highlight their friendly, eager personality.",
    "EntleBucher": "Entlebuchers are medium-sized Swiss herding dogs with a sturdy, muscular frame. Their short, dense coat is tri-colored, complementing their intelligent eyes and strong, agile build.",
    "Eskimo_dog": "American Eskimo Dogs are small to medium-sized with a thick, fluffy double coat that is pure white. Their pointed ears and plumed tails give them an alert, charming appearance.",
    "flat-coated_retriever": "Flat-Coated Retrievers are medium-to-large dogs with a long, glossy black or liver-colored coat. Their elegant frame, strong legs, and cheerful expression reflect their energetic and friendly nature.",
    "French_bulldog": "French Bulldogs are small, muscular dogs with a compact, square body. Their bat-like ears, short muzzle, and smooth coat give them a distinctive, lovable appearance.",
    "German_shepherd": "German Shepherds are large, athletic dogs with a strong, well-proportioned body. Their dense double coat, erect ears, and intelligent eyes convey strength, alertness, and loyalty.",
    "German_short-haired_pointer": "German Shorthaired Pointers are medium-to-large sporting dogs with a lean, muscular build. Their short, dense coat often has liver or liver-and-white markings, highlighting agility and endurance.",
    "giant_schnauzer": "Giant Schnauzers are large, powerful dogs with a robust, square build. Their dense, wiry coat, bushy eyebrows, and beard give them a commanding yet intelligent appearance.",
    "golden_retriever": "Golden Retrievers are medium-to-large dogs with a strong, balanced frame and a dense, water-resistant coat. Their golden hues, expressive eyes, and friendly demeanor make them instantly recognizable.",
    "Gordon_setter": "Gordon Setters are large, elegant sporting dogs with a glossy black coat accented with tan markings. Their long ears, flowing feathering, and athletic build exude grace and endurance.",
    "Great_Dane": "Great Danes are giant dogs with a massive, muscular frame. Their smooth coat comes in various colors, and their large head, expressive eyes, and regal stance convey gentleness and dignity.",
    "Great_Pyrenees": "Great Pyrenees are massive guardian dogs with a thick, white double coat. Their large, sturdy frame, deep chest, and calm, watchful expression reflect strength, protection, and patience.",
     "Greater_Swiss_Mountain_dog": "Greater Swiss Mountain Dogs are large, muscular working dogs with a tri-colored short coat of black, white, and rust. Their sturdy frame, strong legs, and broad head convey power and reliability.",
    "groenendael": "Belgian Sheepdogs (Groenendaels) are medium-to-large dogs with a solid black, long, and dense coat. Their proud carriage, intelligent eyes, and alert ears reflect agility and strong working ability.",
    "Ibizan_hound": "Ibizan Hounds are medium-sized sighthounds with a slender, athletic build. Their large upright ears, long neck, and short coat give them an elegant and alert appearance.",
    "Irish_setter": "Irish Setters are medium-to-large sporting dogs with a lean, graceful frame. Their silky red coat, long feathering on ears, legs, and tail, and friendly expression convey energy and elegance.",
    "Irish_terrier": "Irish Terriers are medium-sized, muscular dogs with a wiry red coat. Their keen eyes, strong jaw, and alert stance give them a confident, spirited look.",
    "Irish_water_spaniel": "Irish Water Spaniels are medium-sized sporting dogs with a curly, waterproof coat. Their long, expressive ears, tight curls, and sturdy frame give a distinctive, smart appearance.",
    "Irish_wolfhound": "Irish Wolfhounds are giant sighthounds with a tall, commanding stature. Their rough coat, long legs, and noble head convey both strength and gentleness.",
    "Italian_greyhound": "Italian Greyhounds are tiny, elegant dogs with a sleek, slender body. Their short coat, graceful limbs, and refined head give them a delicate, agile appearance.",
    "Japanese_spaniel": "Japanese Spaniels are small, elegant dogs with a silky coat and expressive eyes. Their compact body, feathered tail, and alert demeanor make them charming companions.",
    "keeshond": "Keeshonds are medium-sized dogs with a dense double coat of silver and black. Their fox-like face, ruff around the neck, and plumed tail create a striking, friendly appearance.",
    "kelpie": "Australian Kelpies are medium-sized, agile herding dogs with a short, dense coat. Their alert ears, sharp eyes, and lean frame highlight intelligence and stamina.",
    "Kerry_blue_terrier": "Kerry Blue Terriers are medium-sized, muscular dogs with a soft, curly blue-gray coat. Their long head, expressive eyes, and confident stance convey elegance and energy.",
    "komondor": "Komondors are large guardian dogs famous for their corded coat resembling dreadlocks. Their robust frame, broad head, and calm, watchful expression reflect strength and protection.",
    "kuvasz": "Kuvasz dogs are large, powerful guardians with a thick white coat. Their sturdy body, noble head, and protective demeanor give them an imposing yet gentle appearance.",
    "Labrador_retriever": "Labrador Retrievers are medium-to-large dogs with a strong, athletic build. Their short, dense coat, expressive eyes, and friendly face make them instantly recognizable and approachable.",
    "Lakeland_terrier": "Lakeland Terriers are small, compact dogs with a dense, wiry coat. Their sharp eyes, alert ears, and confident stance convey intelligence and courage.",
    "Leonberg": "Leonbergs are giant, majestic dogs with a long, thick double coat. Their lion-like mane, deep chest, and calm, gentle expression create a regal presence.",
    "Lhasa": "Lhasa Apsos are small, sturdy dogs with a long, flowing coat. Their compact body, expressive dark eyes, and proud carriage convey confidence and alertness.",
    "malamute": "Alaskan Malamutes are large, powerful sled dogs with a dense double coat. Their strong frame, bushy tail, and striking facial markings highlight strength and endurance.",
    "malinois": "Belgian Malinois are medium-to-large working dogs with a short fawn coat and black mask. Their athletic build, erect ears, and alert eyes reflect intelligence and agility.",
    "Maltese_dog": "Maltese dogs are small, toy breed companions with a long, silky white coat. Their dark eyes, black nose, and compact body give them an elegant and charming appearance.",
    "Mexican_hairless": "Xoloitzcuintli (Mexican Hairless Dogs) are medium-sized dogs with smooth, hairless skin or very short hair. Their lean frame, alert eyes, and upright ears give a sleek, exotic look.",
    "miniature_pinscher": "Miniature Pinschers are small, muscular dogs with a short, smooth coat. Their upright ears, proud carriage, and compact, agile frame reflect confidence and energy.",
    "miniature_poodle": "Miniature Poodles are small, elegant dogs with a dense, curly coat. Their slender frame, long muzzle, and intelligent expression give them a refined appearance.",
    "miniature_schnauzer": "Miniature Schnauzers are small, sturdy dogs with a wiry coat, bushy eyebrows, and distinctive beard. Their alert expression and compact frame convey spirited intelligence.",
    "Newfoundland": "Newfoundlands are giant, muscular water dogs with a thick, water-resistant coat. Their broad head, deep chest, and gentle eyes reflect strength, calmness, and kindness.",
    "Norfolk_terrier": "Norfolk Terriers are small, compact dogs with a wiry coat and floppy ears. Their alert eyes and confident stance make them lively and charming companions.",
    "Norwegian_elk hound": "Norwegian Elkhounds are medium-sized hunting dogs with a dense, weather-resistant coat. Their strong, compact frame and curled tail convey endurance and agility.",
    "Norwich_terrier": "Norwich Terriers are small, sturdy dogs with a wiry coat and erect ears. Their alert expression and compact body make them agile and spirited.",
    "Old_English_sheepdog": "Old English Sheepdogs are large, shaggy dogs with a dense double coat. Their rounded head, expressive eyes, and strong build create a gentle, iconic appearance.",
    "otterhound": "Otterhounds are large scent hounds with a rough, water-resistant coat. Their strong frame, webbed feet, and expressive eyes reflect endurance and affinity for water.",
    "papillon": "Papillons are small toy dogs with a long, silky coat and large, butterfly-like ears. Their delicate frame and bright, expressive eyes convey elegance and playfulness.",
    "Pekinese": "Pekingese are small, compact dogs with a long, flowing coat and flat face. Their large eyes, short legs, and proud posture give them a regal, charming appearance.",
    "Pembroke": "Pembroke Welsh Corgis are small, sturdy herding dogs with a short, dense coat. Their erect ears, short tail, and intelligent eyes convey energy and loyalty.",
    "Philippine_Forest_dog": "Philippine Forest Dogs are medium-sized, agile dogs with a short, resilient coat. Their lean frame, alert eyes, and athletic stance reflect adaptability and alertness.",
    "Pomeranian": "Pomeranians are tiny, fluffy dogs with a dense double coat. Their fox-like face, plumed tail, and bright, expressive eyes convey lively charm.",
    "pug": "Pugs are small, compact dogs with a wrinkled face, short muzzle, and smooth coat. Their large round eyes, curled tail, and sturdy body give them a charming and expressive look.",
    "redbone": "Redbone Coonhounds are medium-to-large hunting dogs with a short, sleek red coat. Their strong frame, long legs, and alert eyes reflect endurance and focus.",
    "Rhodesian_ridgeback": "Rhodesian Ridgebacks are large, athletic dogs with a short, dense coat. Their distinctive ridge of hair along the back, lean frame, and noble head convey strength and courage.",
    "Rottweiler": "Rottweilers are large, powerful dogs with a short, black-and-tan coat. Their muscular build, broad head, and alert eyes convey confidence and protectiveness.",
    "Saint_Bernard": "Saint Bernards are giant, strong dogs with a dense coat that can be short or long. Their large head, expressive eyes, and deep chest give a gentle, imposing presence.",
    "Saluki": "Salukis are medium-to-large sighthounds with a slender, graceful frame. Their silky coat, long feathered ears, and elegant stance convey agility and nobility.",
    "Samoyed": "Samoyeds are medium-to-large dogs with a thick, fluffy white coat. Their smiling expression, erect ears, and bushy tail convey friendliness and charm.",
    "schipperke": "Schipperkes are small, compact dogs with a dense black coat. Their fox-like face, pointed ears, and confident stance convey curiosity and alertness.",
    "Scotch_terrier": "Scottish Terriers are small, sturdy dogs with a wiry coat and distinctive beard. Their upright ears, deep-set eyes, and dignified stance convey confidence.",
    "Scottish_deerhound": "Scottish Deerhounds are large, lean sighthounds with a rough, wiry coat. Their long legs, narrow head, and gentle expression reflect grace and endurance.",
    "Sealyham_terrier": "Sealyham Terriers are small, sturdy dogs with a wiry coat and compact frame. Their expressive eyes and alert expression make them lively companions.",
    "Shetland_sheepdog": "Shetland Sheepdogs are small-to-medium herding dogs with a long, dense double coat. Their intelligent eyes, alert ears, and elegant stance convey agility and devotion.",
    "Shih-Tzu": "Shih Tzus are small toy dogs with a long, flowing coat. Their short muzzle, large dark eyes, and proud posture give them a regal, charming appearance.",
    "Siberian_husky": "Siberian Huskies are medium-sized sled dogs with a thick double coat. Their erect ears, expressive eyes, and bushy tail convey strength, endurance, and playfulness.",
    "silky_terrier": "Silky Terriers are small dogs with a fine, silky coat. Their compact body, alert expression, and long flowing hair give them elegance and charm.",
    "soft-coated_wheaten_terrier": "Soft-Coated Wheaten Terriers are medium-sized dogs with a soft, wavy coat. Their friendly eyes, sturdy build, and expressive face convey warmth and energy.",
    "Staffordshire_bullterrier": "Staffordshire Bull Terriers are muscular, compact dogs with a short, smooth coat. Their strong jaw, broad head, and confident stance reflect courage and affection.",
    "standard_poodle": "Standard Poodles are medium-to-large dogs with a dense, curly coat. Their elegant posture, long legs, and refined head convey intelligence and grace.",
    "standard_schnauzer": "Standard Schnauzers are medium-sized dogs with a wiry coat, bushy eyebrows, and beard. Their strong, square build and alert expression reflect intelligence and versatility.",
    "Sussex_spaniel": "Sussex Spaniels are medium-sized, low-set sporting dogs with a golden liver-colored coat. Their expressive eyes, floppy ears, and sturdy frame convey a gentle demeanor.",
    "Tibetan_mastiff": "Tibetan Mastiffs are giant, powerful dogs with a dense double coat. Their massive frame, broad head, and watchful eyes reflect strength, independence, and protection.",
    "Tibetan_terrier": "Tibetan Terriers are medium-sized dogs with a long, dense coat. Their compact frame, expressive eyes, and proud carriage convey charm and agility.",
    "toy_poodle": "Toy Poodles are tiny dogs with a dense, curly coat. Their small frame, intelligent expression, and elegant posture make them refined companions.",
    "toy_terrier": "Toy Terriers are small, agile dogs with a smooth coat. Their alert expression, erect ears, and lively frame convey intelligence and energy.",
    "vizsla": "Vizslas are medium-to-large hunting dogs with a short, sleek coat. Their lean, muscular frame, expressive eyes, and noble stance convey athleticism and grace.",
    "Walker_hound": "Walker Hounds are medium-to-large scent hounds with a short coat. Their strong legs, alert expression, and sturdy frame convey endurance and tracking ability.",
    "Weimaraner": "Weimaraners are medium-to-large hunting dogs with a sleek, silver-gray coat. Their muscular build, expressive eyes, and alert ears convey intelligence and athleticism.",
    "Welsh_springer_spaniel": "Welsh Springer Spaniels are medium-sized sporting dogs with a dense, wavy red-and-white coat. Their balanced frame, alert eyes, and friendly expression convey energy and devotion.",
    "West_Highland_white_terrier": "West Highland White Terriers are small, compact dogs with a dense white coat. Their pointed ears, bright eyes, and confident stance convey alertness and charm.",
    "whippet": "Whippets are medium-sized sighthounds with a sleek, lean frame. Their short, smooth coat, long legs, and graceful head convey speed, elegance, and gentle temperament.",
    "wire-haired_fox_terrier": "Wire-Haired Fox Terriers are small, energetic dogs with a dense, wiry coat. Their strong jaw, keen eyes, and alert ears reflect agility and determination.",
    "Yorkshire_terrier": "Yorkshire Terriers are tiny dogs with a long, silky coat. Their compact frame, bright eyes, and erect ears convey intelligence, charm, and liveliness."
}
breed_care = {
    "affenpinscher": "Requires moderate daily exercise, regular brushing for its wiry coat, and social interaction to prevent boredom.",
    "Afghan_hound": "Needs daily long walks, minimal grooming if short clipped, or frequent brushing for long hair, and a calm environment.",
    "African_hunting_dog": "Highly active; requires large outdoor space and socialization within a pack or family to stay happy and mentally stimulated.",
    "Airedale": "Needs daily exercise and mental stimulation, regular brushing, and occasional trimming to maintain coat health.",
    "American_Staffordshire_terrier": "Requires daily walks, interactive play, firm training, and socialization to thrive as a companion.",
    "Appenzeller": "High energy breed; needs daily walks, herding or agility activities, and regular brushing for short coat maintenance.",
    "Aspin": "Adaptable; benefits from daily exercise, balanced nutrition, and routine health checks; low grooming needs.",
    "Australian_terrier": "Moderate exercise, regular grooming to prevent matting, and early socialization for confident behavior.",
    "basenji": "Active indoors and outdoors; needs mental stimulation, occasional brushing, and minimal shedding maintenance.",
    "basset": "Low-impact exercise like walks, regular ear cleaning, and brushing to manage shedding and prevent skin issues.",
    "beagle": "Needs daily walks and play, mental stimulation, and regular brushing to manage short coat shedding.",
    "Bedlington_terrier": "Moderate exercise, regular grooming and trimming, and interactive play to prevent boredom.",
    "Bernese_mountain_dog": "Requires daily walks, cold weather tolerance, regular brushing, and attention to hip/elbow health.",
    "black-and-tan_coonhound": "Active breed; needs long walks, tracking activities, and minimal grooming with attention to ear cleaning.",
    "Blenheim_spaniel": "Moderate exercise, regular brushing to prevent matting, and dental care due to toy size.",
    "bloodhound": "Needs daily walks, scent tracking, strong fencing, and regular ear cleaning to prevent infections.",
    "bluetick": "High-energy breed; requires active play, scent work, and minimal grooming for short coat.",
    "Border_collie": "Extremely high exercise and mental stimulation, agility activities, and moderate brushing for coat maintenance.",
    "Border_terrier": "Daily walks and play, occasional brushing, and firm, consistent training.",
    "borzoi": "Needs daily long walks or running in secure areas, minimal grooming, and soft bedding for large frame.",
    "Boston_bull": "Moderate exercise, short coat care, attention to breathing issues, and dental hygiene.",
    "Bouvier_des_Flandres": "Requires daily exercise, grooming with hand-stripping or trimming, and strong socialization.",
    "boxer": "High-energy; daily exercise and play, short coat brushing, and routine vet care for joints.",
    "Brabancon_griffon": "Low to moderate exercise, regular facial cleaning, and gentle brushing to maintain wiry coat.",
    "briard": "Active breed; daily exercise, frequent brushing, and mental stimulation are essential.",
    "Brittany_spaniel": "Requires daily exercise, agility or hunting activity, and brushing to maintain coat condition.",
    "bull_mastiff": "Moderate exercise, careful with heat, basic grooming, and attention to joints and weight control.",
    "cairn": "Moderate daily walks, brushing to remove dead hair, and mental stimulation through training or games.",
    "Cardigan": "Daily exercise, brushing for coat, and mental stimulation; watch for hip dysplasia in large frames.",
    "Chesapeake_Bay_retriever": "Active; needs swimming or running, regular brushing, and check for ear infections.",
    "Chihuahua": "Short daily walks, gentle play indoors, regular dental care, and occasional coat brushing.",
    "chow": "Moderate exercise, frequent brushing for dense coat, and attention to heat sensitivity.",
    "clumber": "Daily walks, calm environment, regular brushing, and ear cleaning due to droopy ears.",
    "cocker_spaniel": "Moderate exercise, frequent brushing, ear care, and dental hygiene due to small size.",
    "collie": "Daily walks or herding activity, brushing 2-3 times per week, and mental stimulation.",
    "curly-coated_retriever": "Needs swimming or running, moderate brushing for curls, and mental challenges for active mind.",
    "Dandie_Dinmont": "Moderate exercise, regular grooming for coat, and interactive play to prevent boredom.",
    "dhole": "Highly active; needs large outdoor area and social or pack interaction.",
    "dingo": "Active breed; requires secure outdoor space, mental stimulation, and minimal grooming.",
    "Doberman": "High energy; requires daily walks, obedience training, and short coat care.",
    "English_foxhound": "Active; needs daily running or tracking, minimal grooming, and social interaction.",
    "English_setter": "Moderate to high exercise, brushing 2-3 times weekly, and mental stimulation through play.",
    "English_springer": "Requires daily walks, swimming or play, brushing, and ear cleaning to prevent infections.",
    "EntleBucher": "Active; needs daily exercise, brushing, and mental challenges for high-energy breed.",
    "Eskimo_dog": "Needs cold weather exercise, daily brushing, and mental stimulation for intelligent breed.",
    "flat-coated_retriever": "Requires daily play or swimming, brushing weekly, and positive training for obedience.",
    "French_bulldog": "Low to moderate activity, short coat care, monitor for breathing issues and heat sensitivity.",
    "German_shepherd": "High energy; daily walks, obedience training, brushing 2-3 times per week.",
    "German_short-haired_pointer": "Very active; requires running, hunting or play, short coat care, and mental stimulation.",
    "giant_schnauzer": "Needs daily walks, obedience or agility training, and regular grooming to maintain coat.",
    "golden_retriever": "Daily exercise, swimming or running, weekly brushing, and attention to ear care.",
    "Gordon_setter": "Active; needs running, mental stimulation, and brushing to prevent matting.",
    "Great_Dane": "Moderate walks, soft bedding, careful feeding for large breed, short coat brushing.",
    "Great_Pyrenees": "Needs space to roam, daily moderate exercise, brushing weekly, and watch for joint issues.",
    "Greater_Swiss_Mountain_dog": "Moderate exercise, brushing, and careful monitoring of weight and joint health.",
    "groenendael": "Active; requires daily walks, brushing for long black coat, and mental stimulation.",
    "Ibizan_hound": "High energy; needs running or play in secure area, short coat care.",
    "Irish_setter": "Very active; daily walks, brushing to prevent tangles, and mental stimulation.",
    "Irish_terrier": "Moderate exercise, occasional brushing, and consistent training.",
    "Irish_water_spaniel": "Active; daily swimming or play, brushing to maintain curly coat, ear care.",
    "Irish_wolfhound": "Moderate daily walks, soft bedding, basic grooming, attention to joints.",
    "Italian_greyhound": "Short daily walks, indoor play, soft bedding, and gentle grooming.",
    "Japanese_spaniel": "Indoor activity, occasional walks, grooming for long coat, and dental care.",
    "keeshond": "Moderate exercise, brushing 2-3 times per week, and social interaction.",
    "kelpie": "High activity; daily herding or running, brushing, mental stimulation.",
    "Kerry_blue_terrier": "Moderate exercise, regular grooming for curly coat, and interactive play.",
    "komondor": "Needs space to roam, minimal grooming for corded coat, and moderate activity.",
    "kuvasz": "Daily walks, mental stimulation, brushing, and space for movement.",
    "Labrador_retriever": "High energy; daily walks or swimming, short coat care, and play.",
    "Lakeland_terrier": "Moderate activity, grooming for wiry coat, mental stimulation.",
    "Leonberg": "Daily walks, soft bedding, brushing, and attention to joints.",
    "Lhasa": "Indoor activity, regular brushing for long coat, and dental care.",
    "malamute": "Very active; needs running, cold weather play, brushing thick coat.",
    "malinois": "High energy; daily training, obedience, running, and short coat care.",
    "Maltese_dog": "Indoor play, daily brushing for long coat, and dental hygiene.",
    "Mexican_hairless": "Moderate exercise, skin protection from sun, minimal grooming, and warm bedding.",
    "miniature_pinscher": "Short daily walks, interactive play, and occasional brushing.",
    "miniature_poodle": "Daily exercise, brushing for curly coat, and mental stimulation.",
    "miniature_schnauzer": "Moderate walks, brushing or clipping for coat, and interactive training.",
    "Newfoundland": "Daily walks, swimming, brushing for thick coat, and attention to joints.",
    "Norfolk_terrier": "Moderate exercise, grooming for wiry coat, and mental stimulation.",
    "Norwegian_elk hound": "Active; daily walks, brushing, mental stimulation, and secure outdoor space.",
    "Norwich_terrier": "Moderate activity, brushing, and training to prevent stubbornness.",
    "Old_English_sheepdog": "Daily exercise, regular brushing, and mental stimulation.",
    "otterhound": "Active; swimming, daily walks, brushing, and ear cleaning.",
    "papillon": "Indoor activity, short daily walks, brushing, and dental care.",
    "Pekinese": "Minimal activity, indoor play, frequent grooming, and eye care.",
    "Pembroke": "Moderate daily walks, brushing, and mental stimulation.",
    "Philippine_Forest_dog": "Active; daily walks or running, low grooming needs, and mental engagement.",
    "Pomeranian": "Indoor play, short walks, frequent brushing, and dental hygiene.",
    "pug": "Low to moderate activity, monitor weight, short coat care, and breathing observation.",
    "redbone": "Active; daily walks, moderate brushing, and mental stimulation.",
    "Rhodesian_ridgeback": "High energy; long daily walks, minimal grooming, mental stimulation.",
    "Rottweiler": "Moderate activity, short coat care, firm training, and joint monitoring.",
    "Saint_Bernard": "Daily walks, soft bedding, brushing, and attention to weight control.",
    "Saluki": "High-energy; long walks, secure running area, minimal grooming.",
    "Samoyed": "Active; daily walks or running, frequent brushing, and cold weather tolerance.",
    "schipperke": "Moderate exercise, brushing, mental stimulation, and indoor play.",
    "Scotch_terrier": "Moderate daily activity, regular brushing, and consistent training.",
    "Scottish_deerhound": "Moderate activity, soft bedding, minimal grooming, and mental stimulation.",
    "Sealyham_terrier": "Moderate walks, brushing, and playful indoor activity.",
    "Shetland_sheepdog": "Daily walks, brushing 2-3 times per week, and mental engagement.",
    "Shih-Tzu": "Indoor activity, daily brushing, and dental care.",
    "Siberian_husky": "High activity; daily running, cold weather tolerance, and brushing during shedding seasons.",
    "silky_terrier": "Moderate walks, brushing for long coat, and mental stimulation.",
    "soft-coated_wheaten_terrier": "Moderate exercise, brushing for silky coat, and training.",
    "Staffordshire_bullterrier": "Daily walks, interactive play, and short coat care.",
    "standard_poodle": "High energy; daily exercise, brushing for curly coat, and mental stimulation.",
    "standard_schnauzer": "Moderate walks, brushing or clipping, and obedience training.",
    "Sussex_spaniel": "Moderate activity, brushing, ear care, and mental stimulation.",
    "Tibetan_mastiff": "Moderate daily activity, space to roam, minimal grooming, and monitoring weight.",
    "Tibetan_terrier": "Moderate exercise, brushing, mental stimulation, and indoor play.",
    "toy_poodle": "Indoor activity, daily brushing, and mental challenges for intelligence.",
    "toy_terrier": "Indoor play, short walks, occasional brushing, and socialization.",
    "vizsla": "High energy; daily running, mental stimulation, and short coat care.",
    "Walker_hound": "Active; daily walks, scent work, short coat maintenance.",
    "Weimaraner": "High energy; daily exercise, obedience, and short coat care.",
    "Welsh_springer_spaniel": "Moderate activity, brushing, swimming or play, and mental engagement.",
    "West_Highland_white_terrier": "Moderate exercise, brushing, and indoor play.",
    "whippet": "Daily walks or running in secure area, soft bedding, and minimal grooming.",
    "wire-haired_fox_terrier": "Moderate activity, brushing or stripping coat, and mental stimulation.",
    "Yorkshire_terrier": "Indoor play, daily brushing for long coat, and dental care."
}
breed_quick_tips = {
    "affenpinscher": "Socialize early with other dogs and people to prevent small dog syndrome. Provide daily play and short training sessions. Regular brushing and dental care keep them healthy. Monitor for respiratory issues due to short snout.",
    "Afghan_hound": "Require daily exercise to satisfy their high energy. Groom regularly to prevent matting of long coat. Gentle training works best; avoid harsh corrections. Provide soft bedding and ensure safe fencing for running.",
    "African_hunting_dog": "Highly social—needs companionship of other dogs or humans. Require large space and vigorous daily exercise. Diet should be high in protein. Enrichment activities help prevent boredom and destructive behavior.",
    "Airedale": "Provide consistent training and socialization early. Moderate to high exercise daily. Groom weekly to maintain wiry coat. Mental stimulation is essential due to their intelligence. Monitor for hip or joint issues.",
    "American_Staffordshire_terrier": "Engage in structured training and early socialization. Provide daily vigorous exercise and interactive games. Monitor weight to prevent obesity. Positive reinforcement works best. Grooming needs are minimal but check skin folds regularly.",
    "Appenzeller": "High-energy breed—needs both physical and mental exercise. Consistent training and socialization are key. Grooming is moderate. Ideal for experienced owners with active lifestyle. Watch for joint and hip issues.",
    "Aspin": "Social, adaptable, and intelligent. Daily walks and playtime help burn energy. Check for fleas, ticks, and parasites regularly. Offer positive reinforcement in training. Keep them protected from extreme weather conditions.",
    "Australian_terrier": "Provide structured training and early socialization. Regular grooming keeps coat neat. Daily walks and play sessions are essential. Mental stimulation via puzzle toys prevents boredom. Monitor dental health.",
    "basenji": "Exercise daily with off-leash opportunities in safe areas. Intelligent, may get bored—enrichment is critical. Minimal grooming required, but check nails and teeth. Socialization with humans and other dogs is important.",
    "basset": "Gentle, low-energy breed; moderate exercise is enough. Socialize early to prevent stubbornness. Groom regularly to prevent ear infections and maintain coat. Provide joint-friendly bedding. Monitor weight to prevent obesity.",
    "beagle": "Require daily walks and scent work to satisfy hunting instincts. Positive reinforcement training is most effective. Regular grooming keeps coat healthy. Social dogs; supervise interactions with smaller pets. Monitor for obesity.",
    "Bedlington_terrier": "Moderate daily exercise and interactive play recommended. Groom weekly to maintain curly coat. Socialize early to reduce shyness. Mental stimulation prevents destructive behavior. Monitor for liver or kidney issues.",
    "Bernese_mountain_dog": "Require moderate exercise; enjoy outdoor activities. Early socialization and consistent training essential. Grooming weekly to control shedding. Joint supplements may help larger breeds. Monitor weight for healthy growth.",
    "black-and-tan_coonhound": "Exercise daily, especially scent-based activities. Consistent, positive training essential. Minimal grooming needed. Socialize with other dogs to prevent loneliness. Keep in a secure yard due to tracking instinct.",
    "Blenheim_spaniel": "Friendly and social; daily walks and indoor play needed. Regular grooming and ear checks prevent infections. Training is best with gentle reinforcement. Monitor for heart or eye conditions.",
    "bloodhound": "High energy and scent-driven; needs ample outdoor activity. Training can be challenging—patience is key. Groom minimally, focus on ears. Secure yard required due to tracking instinct. Provide mental stimulation regularly.",
    "bluetick": "Require vigorous exercise and mental challenges. Consistent, positive training recommended. Groom weekly to maintain coat. Socialization with family and other pets is important. Check ears and joints regularly.",
    "Border_collie": "Intelligent and energetic; daily physical and mental stimulation mandatory. Ideal for active households. Training should be consistent and challenging. Groom weekly. Prevent boredom with puzzle toys or herding exercises.",
    "Border_terrier": "Moderate daily exercise and play sessions recommended. Consistent, positive training works best. Groom regularly to maintain wiry coat. Early socialization reduces prey drive issues. Mental enrichment is helpful.",
    "borzoi": "Require daily walks and safe off-leash running space. Gentle training recommended. Minimal grooming, but brush regularly. Social and calm; monitor for heart and joint conditions. Provide soft bedding.",
    "Boston_bull": "Daily walks and interactive play essential. Training should be gentle, consistent, and fun. Minimal grooming; focus on dental hygiene. Socialize early with other pets and children. Monitor for respiratory or eye issues.",
    "Bouvier_des_Flandres": "Requires daily exercise and mental challenges. Consistent training needed. Regular grooming to maintain thick coat. Early socialization reduces guarding tendencies. Monitor joint and hip health.",
    "boxer": "High-energy and playful; requires daily vigorous exercise. Gentle, consistent training recommended. Grooming is easy but check teeth and nails. Socialize with other pets and children. Monitor heart health and weight.",
    "Brabancon_griffon": "Needs mental stimulation and daily play. Gentle training works best due to small size. Groom regularly, including facial hair. Socialize early to prevent shyness. Provide warmth and comfort indoors.",
    "briard": "Active breed—daily exercise and mental work essential. Grooming requires regular brushing. Consistent training and early socialization recommended. Monitor joints and provide soft bedding. Ideal for experienced owners.",
    "Brittany_spaniel": "Requires daily exercise and mental stimulation. Positive reinforcement training works best. Groom weekly to maintain coat. Social and friendly; supervise with small children. Monitor for hip and ear health.",
    "bull_mastiff": "Gentle giants needing moderate daily walks. Early socialization is critical. Training should be consistent and calm. Grooming minimal; monitor weight and joints. Provide spacious indoor and outdoor areas.",
    "cairn": "Active, curious breed needing daily play and walks. Training should be consistent with positive reinforcement. Grooming weekly keeps coat healthy. Socialize early and monitor small breed interactions. Prevent boredom with toys.",
    "Cardigan": "Daily walks and moderate play recommended. Consistent training and early socialization essential. Groom weekly. Monitor for joint issues. Provide mental stimulation to keep intelligent breed engaged.",
    "Chesapeake_Bay_retriever": "High-energy breed—needs daily vigorous exercise and swimming if possible. Consistent training and early socialization essential. Groom weekly. Monitor ear health. Provide balanced diet and mental stimulation.",
    "Chihuahua": "Daily play and short walks recommended. Early socialization with humans and other pets is crucial. Gentle training works best. Monitor dental health. Keep warm and avoid rough handling due to small size.",
    "chow": "Independent breed; requires early socialization and consistent training. Moderate exercise daily. Groom thick coat regularly. Monitor weight and dental health. Secure yard is important for outdoor safety.",
    "clumber": "Moderate exercise daily. Gentle and consistent training recommended. Groom regularly and check ears. Socialize early with people and pets. Monitor weight and joint health due to heavy build.",
    "cocker_spaniel": "Daily walks and play sessions required. Grooming for long coat is necessary. Positive reinforcement training works best. Early socialization reduces shyness. Monitor ears and eyes regularly.",
    "collie": "Daily exercise and mental challenges essential. Groom long coat weekly. Gentle training and early socialization recommended. Monitor for eye and joint health. Provide structured play to engage intelligence.",
    "curly-coated_retriever": "Active breed; needs daily walks and swimming opportunities. Groom curly coat weekly. Early training and socialization essential. Monitor joints and provide mental stimulation.",
    "Dandie_Dinmont": "Moderate daily exercise and play. Groom coat regularly. Positive, consistent training recommended. Early socialization helps reduce stubbornness. Monitor dental and joint health.",
    "dhole": "Highly active; needs ample exercise and stimulation. Socialization critical. Provide safe, secure space. Diet rich in protein. Monitor health closely due to wild ancestry.",
    "dingo": "Active and intelligent; daily exercise required. Early socialization essential. Mental stimulation through toys or training. Minimal grooming needed. Provide secure area due to strong prey drive.",
    "Doberman": "Require vigorous daily exercise and mental challenges. Early and consistent training essential. Groom minimally. Socialize regularly. Monitor heart and joint health.",
    "English_foxhound": "Daily exercise and scent activities required. Gentle, consistent training works best. Minimal grooming. Socialize with other dogs. Secure yard important for tracking instincts.",
    "English_setter": "Daily walks and play recommended. Groom weekly to maintain coat. Positive reinforcement training. Socialize early with humans and pets. Monitor hips and ears.",
    "English_springer": "Active breed; daily walks and play essential. Groom coat weekly. Early socialization and training. Provide mental stimulation. Monitor ears and joints.",
    "EntleBucher": "Daily exercise and mental challenges needed. Early socialization important. Groom weekly. Training consistent with positive reinforcement. Monitor joints and overall health.",
    "Eskimo_dog": "Daily exercise required. Grooming fluffy coat regularly. Socialization with people and dogs important. Early training recommended. Monitor teeth and joints.",
    "flat-coated_retriever": "Daily exercise and swimming opportunities beneficial. Groom weekly. Positive training essential. Socialization important. Monitor diet and joint health.",
    "French_bulldog": "Short daily walks; avoid overheating. Gentle, consistent training recommended. Groom minimally; monitor skin folds. Socialize early. Watch respiratory health.",
    "German_shepherd": "High-energy; daily walks, runs, and mental stimulation required. Early training and socialization crucial. Groom weekly. Monitor joints and hips.",
    "German_short-haired_pointer": "Daily vigorous exercise and hunting games recommended. Early training and socialization essential. Groom minimally. Monitor diet and joints.",
    "giant_schnauzer": "High-energy; requires physical and mental exercise daily. Early training essential. Groom coat regularly. Socialize with family and dogs. Monitor joints.",
    "golden_retriever": "Daily walks, play, and swimming opportunities. Groom weekly. Early socialization and consistent training. Monitor weight and dental health. Mental stimulation important.",
    "Gordon_setter": "Daily walks and hunting activities recommended. Groom weekly. Positive reinforcement training. Socialize early. Monitor joints and ears.",
    "Great_Dane": "Gentle walks daily; avoid over-exercising growing puppies. Early socialization important. Groom minimally. Monitor joints and heart health. Provide spacious living space.",
    "Great_Pyrenees": "Moderate daily exercise. Groom thick coat regularly. Early socialization and gentle training. Monitor weight and joints. Provide secure yard.",
    "Greater_Swiss_Mountain_dog": "Daily walks and play. Groom weekly. Early socialization and training. Monitor joints and weight. Provide mental stimulation and structured play.",
    "groenendael": "Daily exercise and mental stimulation essential. Groom long coat weekly. Early training recommended. Socialize with family and other pets. Monitor joints.",
    "Ibizan_hound": "High-energy; requires daily walks or running. Gentle training and socialization early. Groom minimally. Provide secure yard. Mental stimulation helps prevent boredom.",
    "Irish_setter": "Daily vigorous exercise required. Groom coat regularly. Early socialization and positive training essential. Monitor joints and diet. Provide mental stimulation.",
    "Irish_terrier": "Moderate daily exercise. Positive, consistent training. Early socialization important. Groom weekly. Monitor dental health. Mental stimulation prevents boredom.",
    "Irish_water_spaniel": "Daily exercise, swimming encouraged. Groom curly coat weekly. Early training and socialization recommended. Monitor joints and teeth. Provide mental enrichment.",
    "Irish_wolfhound": "Moderate daily walks. Early socialization crucial. Gentle training recommended. Groom weekly. Monitor heart and joints. Provide large, comfortable living space.",
    "Italian_greyhound": "Short daily walks and play. Gentle training with positive reinforcement. Groom minimally. Socialize early. Monitor for fragile bones.",
    "Japanese_spaniel": "Daily play and short walks. Groom coat regularly. Early socialization and gentle training. Monitor dental health. Provide warm, safe living space.",
    "keeshond": "Daily walks and playtime. Groom thick coat weekly. Early socialization and training. Monitor teeth and weight. Provide mental stimulation.",
    "kelpie": "High-energy; daily exercise and mental challenges required. Early training essential. Groom minimally. Monitor joints and diet. Provide secure yard.",
    "Kerry_blue_terrier": "Daily exercise and play recommended. Groom weekly. Early socialization and training. Monitor dental and skin health. Mental stimulation important.",
    "komondor": "Moderate daily exercise. Groom corded coat carefully. Early socialization and training crucial. Monitor weight and joints. Provide secure yard.",
    "kuvasz": "Daily exercise and mental stimulation needed. Groom weekly. Early socialization and training recommended. Monitor weight and joints. Provide secure yard.",
    "Labrador_retriever": "Daily walks, play, and swimming. Groom weekly. Early training and socialization essential. Monitor weight and joints. Provide mental enrichment.",
    "Lakeland_terrier": "Moderate daily exercise and play. Groom weekly. Early socialization and positive training. Monitor teeth. Mental stimulation important.",
    "Leonberg": "Daily walks and moderate exercise. Groom thick coat weekly. Early socialization and gentle training. Monitor joints and weight. Provide large living space.",
    "Lhasa": "Daily walks and indoor play. Groom long coat regularly. Early socialization and training. Monitor dental health. Provide warm environment.",
    "malamute": "High-energy; daily long walks or runs. Groom thick coat weekly. Early socialization and training. Monitor joints and weight. Provide secure yard.",
    "malinois": "Vigorous daily exercise required. Early socialization and consistent training essential. Groom minimally. Monitor joints and weight. Provide mental stimulation.",
    "Maltese_dog": "Daily short walks and play. Groom coat regularly. Early socialization and gentle training. Monitor dental health. Provide safe, warm environment.",
    "Mexican_hairless": "Daily walks and indoor play. Minimal grooming, but skin care important. Early socialization and training recommended. Monitor teeth and temperature sensitivity.",
    "miniature_pinscher": "Daily play and short walks. Early socialization and positive training essential. Groom minimally. Monitor weight and dental health. Mental stimulation recommended.",
    "miniature_poodle": "Daily walks and mental stimulation needed. Groom coat regularly. Early training and socialization important. Monitor dental health. Positive reinforcement recommended.",
    "miniature_schnauzer": "Daily exercise and play. Groom regularly, including beard and eyebrows. Early socialization and consistent training. Monitor weight and teeth. Mental stimulation important.",
    "Newfoundland": "Moderate daily exercise. Groom coat weekly. Early socialization and gentle training recommended. Monitor joints and weight. Provide large living space and soft bedding.",
    "Norfolk_terrier": "Daily walks and play. Groom coat weekly. Early socialization and consistent training. Monitor dental health. Mental stimulation helps prevent boredom.",
    "Norwegian_elk hound": "Daily walks and play sessions. Early socialization essential. Moderate grooming. Monitor joints and teeth. Provide secure yard due to hunting instincts.",
    "Norwich_terrier": "Daily exercise and interactive play. Groom coat weekly. Early socialization and training recommended. Monitor teeth. Mental stimulation important.",
    "Old_English_sheepdog": "Moderate daily exercise. Groom thick coat regularly. Early socialization and training essential. Monitor joints and weight. Provide mental stimulation.",
    "otterhound": "High-energy; daily walks and swimming recommended. Groom coat regularly. Early socialization and positive training essential. Monitor weight and joints.",
    "papillon": "Daily play and short walks. Groom coat regularly. Early socialization and training. Monitor dental health. Provide mental stimulation.",
    "Pekinese": "Short daily walks and indoor play. Groom facial folds regularly. Early socialization and gentle training. Monitor respiratory and eye health. Provide safe environment.",
    "Pembroke": "Daily walks and play. Groom coat minimally. Early socialization and consistent training. Monitor weight and joints. Mental stimulation important.",
    "Philippine_Forest_dog": "Daily exercise and mental stimulation essential. Early socialization and positive training. Groom minimally. Monitor for parasites and weight. Provide secure yard.",
    "Pomeranian": "Daily play and short walks. Groom coat regularly. Early socialization and training recommended. Monitor teeth and weight. Provide warm environment.",
    "pug": "Short daily walks and play. Groom minimally, focus on skin folds. Early socialization and gentle training. Monitor respiratory health. Mental stimulation important.",
    "redbone": "Daily walks and scent work recommended. Early training and socialization essential. Groom coat weekly. Monitor joints and weight. Provide secure yard.",
    "Rhodesian_ridgeback": "Daily vigorous exercise required. Early training and socialization essential. Groom minimally. Monitor joints, weight, and heart. Provide secure yard.",
    "Rottweiler": "Daily walks and mental stimulation required. Early socialization and consistent training essential. Groom minimally. Monitor joints and weight. Provide secure space.",
    "Saint_Bernard": "Moderate daily walks. Groom coat regularly. Early socialization and gentle training. Monitor joints and weight. Provide spacious indoor area.",
    "Saluki": "Daily walks and running opportunities. Gentle training and early socialization recommended. Groom minimally. Monitor joints and weight. Provide secure yard.",
    "Samoyed": "Daily walks and play required. Groom coat regularly. Early socialization and consistent training. Monitor teeth and joints. Provide mental stimulation.",
    "schipperke": "Daily exercise and play. Groom weekly. Early socialization and consistent training. Monitor teeth and weight. Mental stimulation important.",
    "Scotch_terrier": "Moderate daily walks and play sessions. Groom coat regularly. Early socialization and consistent training. Monitor dental health. Mental stimulation helpful.",
    "Scottish_deerhound": "Daily walks and play required. Gentle training and early socialization. Groom coat weekly. Monitor joints and weight. Provide spacious living space.",
    "Sealyham_terrier": "Daily walks and interactive play. Groom coat weekly. Early socialization and consistent training. Monitor dental health. Mental stimulation recommended.",
    "Shetland_sheepdog": "Daily exercise and mental challenges. Groom long coat regularly. Early training and socialization essential. Monitor joints. Provide structured play.",
    "Shih-Tzu": "Daily short walks and indoor play. Groom coat regularly. Early socialization and gentle training. Monitor dental health. Provide warm environment.",
    "Siberian_husky": "High-energy; daily vigorous exercise required. Early socialization and consistent training. Groom coat weekly. Monitor joints and weight. Provide secure yard.",
    "silky_terrier": "Daily play and short walks. Groom coat regularly. Early socialization and consistent training. Monitor dental health. Mental stimulation recommended.",
    "soft-coated_wheaten_terrier": "Daily exercise and play. Groom coat regularly. Early socialization and consistent training. Monitor dental health. Mental stimulation helps prevent boredom.",
    "Staffordshire_bullterrier": "Daily walks and vigorous play. Early socialization and training essential. Groom minimally. Monitor joints and weight. Mental stimulation important.",
    "standard_poodle": "Daily walks and mental stimulation required. Groom coat regularly. Early socialization and consistent training. Monitor joints and weight. Provide puzzle toys.",
    "standard_schnauzer": "Daily exercise and mental challenges. Groom coat regularly. Early socialization and training essential. Monitor joints. Provide structured play.",
    "Sussex_spaniel": "Moderate daily exercise. Groom coat regularly. Early socialization and gentle training. Monitor weight. Mental stimulation helpful.",
    "Tibetan_mastiff": "Moderate daily walks. Groom coat regularly. Early socialization and consistent training. Monitor joints and weight. Provide secure yard.",
    "Tibetan_terrier": "Daily play and short walks. Groom coat regularly. Early socialization and gentle training. Monitor dental health. Mental stimulation recommended.",
    "toy_poodle": "Daily walks and mental stimulation. Groom coat regularly. Early socialization and training essential. Monitor joints and weight. Provide puzzle toys.",
    "toy_terrier": "Daily play and short walks. Early socialization and consistent training. Groom minimally. Monitor dental health. Mental stimulation important.",
    "vizsla": "Daily vigorous exercise required. Early socialization and training essential. Groom minimally. Monitor joints and weight. Provide secure yard.",
    "Walker_hound": "Daily walks and scent activities recommended. Early training and socialization essential. Groom minimally. Monitor joints and weight. Provide secure yard.",
    "Weimaraner": "High-energy; daily vigorous exercise required. Early training and socialization essential. Groom minimally. Monitor joints and weight. Provide mental stimulation.",
    "Welsh_springer_spaniel": "Daily walks and play required. Groom coat weekly. Early training and socialization essential. Monitor joints and weight. Mental stimulation recommended.",
    "West_Highland_white_terrier": "Daily walks and play sessions. Groom coat regularly. Early socialization and consistent training. Monitor dental health. Mental stimulation helpful.",
    "whippet": "Daily exercise and safe off-leash running recommended. Gentle training and early socialization essential. Groom minimally. Monitor weight. Provide soft bedding.",
    "wire-haired_fox_terrier": "Daily walks and play. Groom wiry coat regularly. Early socialization and consistent training. Monitor dental health. Mental stimulation recommended.",
    "Yorkshire_terrier": "Daily play and short walks. Groom coat regularly. Early socialization and training essential. Monitor dental health. Provide warm and safe environment."
}

# -----------------------------
# HELPERS
# -----------------------------
def expand_description(
    breed_key: str, 
    short_desc: str, 
    breed_temperaments: dict, 
    breed_appearance: dict,
    breed_care: dict,
    breed_quick_tips: dict
) -> str:
    if not breed_key or not isinstance(breed_key, str):
        breed_key = "Unknown_Breed"
    pretty_name = breed_key.replace("_", " ").replace("-", " ").title()
    special_history = ""
    lower = breed_key.lower()
    if "aspin" in lower or "philippine" in lower:
        special_history = (
            f"{pretty_name} traces its roots to the Philippines' streets and rural areas — "
            "a resilient mixed-heritage companion adapted to diverse conditions."
        )
    history = (
        f"<b>History & Origins ({pretty_name}):</b> {short_desc} "
        + (special_history + " " if special_history else "")
        + f"Historically, {pretty_name} developed traits suited to its original role—"
          "whether hunting, guarding, herding, or companionship."
    )
    appearance_desc = breed_appearance.get(
        breed_key, 
        f"{pretty_name} typically shows characteristic physical features such as coat texture, body build, and proportions."
    )
    appearance = f"<b>Appearance ({pretty_name}):</b> {appearance_desc}"
    temperament_desc = breed_temperaments.get(breed_key, 
        "This breed shows a range of temperaments depending on lineage and environment."
    )
    temperament = (
        f"<b>Temperament & Behavior ({pretty_name}):</b> {temperament_desc} "
        f"{pretty_name} often displays behaviors shaped by its historical purpose, including trainability, "
        "social tendencies, and energy levels."
    )
    care_desc = breed_care.get(
        breed_key, 
        f"Grooming, exercise, and overall care requirements differ for {pretty_name}. "
        "Regular vet checkups and a balanced diet help maintain overall health."
    )
    care = f"<b>Care & Needs ({pretty_name}):</b> {care_desc}"
    quick_desc = breed_quick_tips.get(
        breed_key,
        "Socialize early, provide consistent training, tailor exercise needs, "
        "and monitor for breed-associated health tendencies."
    )
    quick = f"<b>Quick Tips for {pretty_name} Owners:</b> {quick_desc}"
    combined = (
        "<para spaceb=6>"
        + history + "<br/><br/>"
        + appearance + "<br/><br/>"
        + temperament + "<br/><br/>"
        + care + "<br/><br/>"
        + quick
        + "</para>"
    )
    return combined

def preprocess_image(file_stream):
    img = Image.open(file_stream).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0).astype(np.float32)
    return arr

@app.route("/reports/<filename>")
def serve_report(filename):
    report_dir = os.path.join(STATIC_DIR, "reports")
    return send_from_directory(report_dir, filename)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "PawPrint backend is running"}), 200

# -----------------------------
# PREDICT endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "no image file"}), 400

        file = request.files["image"]
        x = preprocess_image(file.stream)

        preds_dict = infer(tf.constant(x))
        preds = list(preds_dict.values())[0].numpy()[0]

        top_indices = preds.argsort()[-3:][::-1]

        results = []
        for idx in top_indices:
            breed = idx_to_class[int(idx)]
            conf = float(preds[int(idx)])

            # ✅ Updated static image URL generation
            sample_path = os.path.join(STATIC_DIR, "breed_examples", f"{breed}.jpg")
            sample_url = None
            if os.path.exists(sample_path):
                sample_url = url_for('static', filename=f'breed_examples/{breed}.jpg', _external=True)

            short_desc = breed_descriptions.get(breed, "This breed is known for its unique traits.")

            long_desc = expand_description(
                breed,
                short_desc,
                breed_temperaments,
                breed_appearance,
                breed_care,
                breed_quick_tips
            )

            results.append({
                "breed": breed,
                "confidence": conf,
                "example_image": sample_url,
                "short_description": short_desc,
                "long_description": long_desc,
                "temp_file_name": file.filename
            })

        return jsonify({"predictions": results})

    except Exception as e:
        app.logger.exception("Predict error")
        return jsonify({"error": str(e)}), 500

# -----------------------------
# GENERATE PDF endpoint
# -----------------------------
@app.route("/generate_pdf", methods=["POST"])
@cross_origin()
def generate_pdf():
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.pagesizes import legal
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    breed_raw = request.form.get("breed", "Unknown_Breed")
    breed_display = breed_raw.replace("_", " ").replace("-", " ").title()
    breed = breed_raw
    confidence = float(request.form.get("confidence", 0.0))
    uploaded_file = request.files.get("image")

    short_desc = breed_descriptions.get(breed, "A well-loved companion breed.")

    description = expand_description(
        breed,
        short_desc,
        breed_temperaments,
        breed_appearance,
        breed_care,
        breed_quick_tips
    )

    img_path = None
    if uploaded_file and uploaded_file.filename:
        upload_dir = os.path.join(STATIC_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        safe_filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", uploaded_file.filename)
        img_path = os.path.join(upload_dir, safe_filename)
        uploaded_file.save(img_path)

    report_dir = os.path.join(STATIC_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_breed_name = re.sub(r"[^a-zA-Z0-9_-]", "_", breed)
    output_path = os.path.join(report_dir, f"{safe_breed_name}_report.pdf")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=legal,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    FRAME_WIDTH = doc.width
    story = []

    # Header
    header_data = [[breed_display], [""]]
    header_table = Table(header_data, colWidths=[FRAME_WIDTH], rowHeights=[30, 15])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.black),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 22),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(header_table)

    # Info Table
    info_data = [
    [" Predicted Breed", f" {breed_display}"],
    [" Confidence", f" {confidence:.4f}"],
    [" Generated", f" {timestamp}"],
    ]

    info_table = Table(info_data, colWidths=[FRAME_WIDTH * 0.35, FRAME_WIDTH * 0.65])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f5f7fa")),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 12),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3 * inch))

    # Uploaded image
    if img_path:
        img = RLImage(img_path)
        img._restrictSize(FRAME_WIDTH * 0.8, 350)
        story.append(img)
        story.append(Spacer(1, 0.3 * inch))

    # Description
    desc_style = ParagraphStyle(
        name="Description",
        fontSize=11,
        leading=16,
        alignment=4,
        spaceBefore=10
    )
    story.append(Paragraph(description, desc_style))

    # Footer
    footer_table = Table([[ "\U0001F43E PawPrint", f"Generated: {timestamp}" ]], colWidths=[FRAME_WIDTH * 0.5, FRAME_WIDTH * 0.5])
    footer_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (0,0), 'LEFT'),
        ('ALIGN', (1,0), (1,0), 'RIGHT'),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Oblique'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.grey),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(Spacer(1, 0.5 * inch))
    story.append(footer_table)

    # Build PDF
    doc.build(story)

    # Return the URL via the new dynamic route
    pdf_url = url_for("serve_report", filename=f"{safe_breed_name}_report.pdf", _external=True)
    return jsonify({"pdf_url": pdf_url})



# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))