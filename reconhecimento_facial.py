import face_recognition
from PIL import Image, ImageDraw
import numpy as np

## Pessoas a ser reconhecida na foto

danilo_image = face_recognition.load_image_file("img/img/Danilo.jpg")
danilo_face_encoding = face_recognition.face_encodings(danilo_image)[0]

dudu_image = face_recognition.load_image_file("img/img/Dudu.jpg")
dudu_face_encoding = face_recognition.face_encodings(dudu_image)[0]

gustavo_gomez_image = face_recognition.load_image_file("img/img/GustavoGomez.jpg")
gustavo_gomez_face_encoding = face_recognition.face_encodings(gustavo_gomez_image)[0]

luan_image = face_recognition.load_image_file("img/img/Luan.jpg")
luan_face_encoding = face_recognition.face_encodings(luan_image)[0]

marcos_rocha_image = face_recognition.load_image_file("img/img/MarcosRocha.jpg")
marcos_rocha_face_encoding = face_recognition.face_encodings(marcos_rocha_image)[0]

piquerez_image = face_recognition.load_image_file("img/img/Piquerez.jpg")
piquerez_face_encoding = face_recognition.face_encodings(piquerez_image)[0]

rony_image = face_recognition.load_image_file("img/img/Rony.jpg")
rony_face_encoding = face_recognition.face_encodings(rony_image)[0]

scarpa_image = face_recognition.load_image_file("img/img/Scarpa.jpg")
scarpa_face_encoding = face_recognition.face_encodings(scarpa_image)[0]

veiga_image = face_recognition.load_image_file("img/img/Veiga.jpg")
veiga_face_encoding = face_recognition.face_encodings(veiga_image)[0]

weverton_image = face_recognition.load_image_file("img/img/Weverton.jpg")
weverton_face_encoding = face_recognition.face_encodings(weverton_image)[0]

ze_rafael_image = face_recognition.load_image_file("img/img/ZeRafael.jpg")
ze_rafael_face_encoding = face_recognition.face_encodings(ze_rafael_image)[0]

known_face_encodings = [
    danilo_face_encoding,
    dudu_face_encoding,
    gustavo_gomez_face_encoding,
    luan_face_encoding,
    marcos_rocha_face_encoding,
    piquerez_face_encoding,
    rony_face_encoding,
    scarpa_face_encoding,
    veiga_face_encoding,
    weverton_face_encoding,
    ze_rafael_face_encoding
]

known_face_names = [
    "Danilo",
    "Dudu",
    "Gustavo Gomez",
    "Luan",
    "Marcos Rocha",
    "Piquerez",
    "Rony",
    "Gustavo Scarpa",
    "Raphael Veiga",
    "Weverton",
    "Zé Rafael"
]


# Load an image with an unknown face
unknown_image = face_recognition.load_image_file("img/img/fototeste3.jpg")

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(unknown_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # If a match was found in known_face_encodings, just use the first one.
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
pil_image.show()
