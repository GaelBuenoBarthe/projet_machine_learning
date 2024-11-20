import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient
import math

# Function to draw a polygon with a unique number
def draw_polygon_with_number(draw, points, number, fill_color="yellow", outline_color="black"):
    draw.polygon(points, fill=fill_color, outline=outline_color)
    centroid = polygon_centroid(points)
    font = ImageFont.truetype("arial.ttf", 40)  # Adjustable size
    draw.text(centroid, str(number), fill="red", font=font)

# Function to calculate the distance between two points
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to sort points to form a polygon
def sort_points_for_polygon(points):
    if not points:
        return []
    ordered_points = []
    current_point = points.pop(0)
    ordered_points.append(current_point)
    while points:
        closest_point = min(points, key=lambda p: distance(current_point, p))
        ordered_points.append(closest_point)
        points.remove(closest_point)
        current_point = closest_point
    return ordered_points

# Function to calculate the centroid of a polygon
def polygon_centroid(points):
    n = len(points)
    area = 0
    Cx = 0
    Cy = 0

    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        cross_product = x0 * y1 - x1 * y0
        area += cross_product
        Cx += (x0 + x1) * cross_product
        Cy += (y0 + y1) * cross_product

    area = abs(area) / 2
    if area != 0:
        Cx = Cx / (6 * area)
        Cy = Cy / (6 * area)
    else:
        Cx, Cy = sum(x for x, y in points) / n, sum(y for x, y in points) / n

    return (Cx, Cy)

# Main function for the Streamlit page
def nail_page():
    st.header("Bienvenue")
    st.caption("Bienvenue dans la détection d'ongles")

    c1, _ = st.columns([4, 2], vertical_alignment="center")

    with c1:
        options = ["Modèle de GAEL", "Modèle basique de Fabrice", "Modèle avancé de Fabrice", "Superposer les modèles", "Nouveau modèle"]
        selected_option = st.selectbox("Choisissez un modèle :", options)
        if selected_option == "Nouveau modèle":
            input1 = st.text_input("Entrez l'api_key de Robotflow :", key="premier bouton")
            input2 = st.text_input("Entrez le model_id de Robotflow :", key="second bouton")

        choice = st.radio(
            "Choisissez un seuil en pourcentage pour la confiance de prédiction :",
            ("Pas de seuil", "50%", "60%", "70%", "80%", "85%", "90%")
        )

        if choice == "Pas de seuil":
            seuil = 0
        elif choice == "50%":
            seuil = 0.5
        elif choice == "60%":
            seuil = 0.6
        elif choice == "70%":
            seuil = 0.7
        elif choice == "80%":
            seuil = 0.8
        elif choice == "85%":
            seuil = 0.85
        elif choice == "90%":
            seuil = 0.9

        uploaded_file = st.file_uploader("Choisissez une image", type=["png", "jpg", "jpeg"])

    col1, col2 = st.columns(2, vertical_alignment="center")

    if uploaded_file is not None:
        ima = Image.open(uploaded_file)
        image = ima
        im_non_traitee = image.save("geeks.jpg")

        with col1:
            image = ImageOps.exif_transpose(image)
            target_size = (340, 340)
            ratio = min(target_size[0] / image.width, target_size[1] / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            new_image = Image.new('L', target_size, 'black')
            left = (target_size[0] - new_size[0]) // 2
            top = (target_size[1] - new_size[1]) // 2
            new_image.paste(image, (left, top))
            processed_image_path = 'IMG_2075_processed.jpg'
            new_image.save(processed_image_path)
            new_image = new_image.convert("L")
            processed_image_path = "IMG_2075_processed.jpg"
            new_image.save(processed_image_path)

            liste2 = []  # Initialize liste2
            counter = 1  # Initialize counter

            try:
                if selected_option == "Modèle de GAEL":
                    GAEL = InferenceHTTPClient(
                        api_url="https://detect.roboflow.com",
                        api_key="C1QXXjGrgpWRBq8uQfS7"
                    )
                    result = GAEL.infer(processed_image_path, model_id="nails-diginamic-7syht/1")
                    result_non_traite = GAEL.infer("geeks.jpg", model_id="nails-diginamic-7syht/1")
                    if 'predictions' in result:
                        for prediction in result['predictions']:
                            confidence = prediction.get('confidence', 'N/A')
                            if confidence > seuil:
                                st.write(f"Confiance de la prédiction Gaël : {confidence}")
                                if 'points' in prediction:
                                    points = prediction.get('points')
                                    for point in points:
                                        x = point['x']
                                        y = point['y']
                                        liste2.append((x, y))

                                    ordered_points = sort_points_for_polygon(liste2)
                                    draw = ImageDraw.Draw(new_image)
                                    draw_polygon_with_number(draw, ordered_points, counter)
                                else:
                                    print("Aucune annotation de polygone trouvée.")
                    else:
                        st.write("Aucune prédiction trouvée.")
                elif selected_option == "Modèle basique de Fabrice":
                    CLIENT = InferenceHTTPClient(
                        api_url="https://detect.roboflow.com",
                        api_key="D7dNUce8UyrrxqFcplzH"
                    )
                    result = CLIENT.infer(processed_image_path, model_id="nails-diginamic-el24e/2")
                    result_non_traite = CLIENT.infer("geeks.jpg", model_id="nails-diginamic-el24e/2")
                elif selected_option == "Modèle avancé de Fabrice":
                    CLIENT = InferenceHTTPClient(
                        api_url="https://detect.roboflow.com",
                        api_key="D7dNUce8UyrrxqFcplzH"
                    )
                    result = CLIENT.infer(processed_image_path, model_id="nails-diginamic-el24e/3")
                    result_non_traite = CLIENT.infer("geeks.jpg", model_id="nails-diginamic-el24e/3")
                elif selected_option == "Nouveau modèle" and input1 and input2:
                    CLIENT = InferenceHTTPClient(
                        api_url="https://detect.roboflow.com",
                        api_key=input1
                    )
                    result = CLIENT.infer(processed_image_path, model_id=input2)
                    result_non_traite = CLIENT.infer("geeks.jpg", model_id=input2)

            except Exception as e:
                st.write(f"Erreur lors de l'inférence block 1 : {e}, si Nouveau modèle, entrer les deux valeurs d'input")

            if selected_option == "Superposer les modèles":
                try:
                    GAEL = InferenceHTTPClient(
                        api_url="https://detect.roboflow.com",
                        api_key="C1QXXjGrgpWRBq8uQfS7"
                    )
                    result2 = GAEL.infer(processed_image_path, model_id="nails-diginamic-7syht/1")
                    liste2 = []
                    counter2 = 0
                    if 'predictions' in result2:
                        for prediction2 in result2['predictions']:
                            confidence2 = prediction2.get('confidence', 'N/A')
                            if confidence2 > seuil:
                                st.write(f"Confiance de la prédiction Gaël : {confidence2}")
                                if 'points' in prediction2:
                                    points = prediction2.get('points')
                                    for point in points:
                                        x = point['x']
                                        y = point['y']
                                        liste2.append((x, y))

                                    ordered_points2 = sort_points_for_polygon(liste2)
                                    draw = ImageDraw.Draw(new_image)
                                    font = ImageFont.load_default()

                                    visited = set()
                                    polygons2 = []
                                    for point in ordered_points2:
                                        if point not in visited:
                                            nearest = min(ordered_points2, key=lambda p: distance(point, p))
                                            polygon = [point, nearest]
                                            polygons2.append(polygon)
                                            visited.update(polygon)
                                    counter2 = 1
                                    draw_polygon_with_number(draw, ordered_points2, counter2)

                                else:
                                    print("Aucune annotation de polygone trouvée.")
                    else:
                        st.write("Aucune prédiction trouvée.")
                except Exception as e:
                    st.write(f"Erreur lors de l'inférence block 2 : {e}")

            try:
                ordered_points = sort_points_for_polygon(liste2)

                GAEL = InferenceHTTPClient(
                    api_url="https://detect.roboflow.com",
                    api_key="C1QXXjGrgpWRBq8uQfS7"
                )
                result3 = GAEL.infer("geeks.jpg", model_id="nails-diginamic-7syht/1")
                liste3 = []
                if 'predictions' in result3:
                    for prediction3 in result3['predictions']:
                        confidence3 = prediction3.get('confidence', 'N/A')
                        if confidence3 > seuil:
                            st.write(f"Confiance de la prédiction Gaël, image non traitée : {confidence3}")
                            if 'points' in prediction3:
                                points = prediction3.get('points')
                                for point in points:
                                    x = point['x']
                                    y = point['y']
                                    liste3.append((x, y))

                                ordered_points3 = sort_points_for_polygon(liste3)
                                draw = ImageDraw.Draw(ima)
                                draw_polygon_with_number(draw, ordered_points3, counter)
                            else:
                                print("Aucune annotation de polygone trouvée.")
                else:
                    st.write("Aucune prédiction trouvée.")
            except Exception as e:
                st.write(f"Erreur lors de l'inférence block 3 : {e}")

            try:
                CLIENT = InferenceHTTPClient(
                    api_url="https://detect.roboflow.com",
                    api_key="D7dNUce8UyrrxqFcplzH"
                )
                result4 = CLIENT.infer(processed_image_path, model_id="nails-diginamic-el24e/3")
                liste4 = []
                if 'predictions' in result4:
                    for prediction4 in result4['predictions']:
                        confidence4 = prediction4.get('confidence', 'N/A')
                        if confidence4 > seuil:
                            st.write(f"Confiance de la prédiction Fabrice, image non traitée : {confidence4}")
                            if 'points' in prediction4:
                                points = prediction4.get('points')
                                for point in points:
                                    x = point['x']
                                    y = point['y']
                                    liste4.append((x, y))

                                ordered_points4 = sort_points_for_polygon(liste4)
                                draw = ImageDraw.Draw(ima)
                                draw_polygon_with_number(draw, ordered_points4, counter)
                                counter += 1
                            else:
                                print("Aucune annotation de polygone trouvée.")
                else:
                    st.write("Aucune prédiction trouvée.")
            except Exception as e:
                st.write(f"Erreur lors de l'inférence block 4 : {e}")

        with col2:
            st.image(ima, caption="Image non traitée", use_container_width=True)
            st.image(new_image, caption="Image traitée avec prédictions", use_container_width=True)