import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
import pandas as pd
import math

def draw_polygon_pil(points, image_size=(300, 300), fill_color="black", outline_color="black"):
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)
    draw.polygon(points, fill=fill_color, outline=outline_color)
    image.show()

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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

def polygon_barycenter(points):
    n = len(points)
    Cx = sum(x for x, y in points) / n
    Cy = sum(y for x, y in points) / n
    return (Cx, Cy)

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
        return polygon_barycenter(points)
    return (Cx, Cy)

def nail_page():
    st.header("Bienvenue")
    st.caption("Bienvenue dans la détection d'ongle")

    c1, _ = st.columns([4, 2])

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

        seuil = 0
        if choice == "50%":
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

    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        ima = Image.open(uploaded_file)
        image = ima
        im_non_traitee = image.save("sections/nailsdetection/pictures/geeks.jpg")

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
            processed_image_path = 'sections/nailsdetection/pictures/IMG_2075_processed.jpg'
            new_image.save(processed_image_path)
            new_image = new_image.convert("L")
            new_image.save(processed_image_path)

            # Placeholder for inference logic
            result = {"predictions": [{"confidence": 0.8, "points": [{"x": 50, "y": 50}, {"x": 100, "y": 50}, {"x": 100, "y": 100}, {"x": 50, "y": 100}]}]}
            result_non_traite = {"predictions": [{"confidence": 0.8, "points": [{"x": 50, "y": 50}, {"x": 100, "y": 50}, {"x": 100, "y": 100}, {"x": 50, "y": 100}]}]}

            if selected_option == "Superposer les modèles":
                counter = 1
                try:
                    font_path = "sections/nailsdetection/font/Roboto-Bold.ttf"
                    font = ImageFont.truetype(font_path, size=30)
                    result2 = result
                    liste2 = []
                    if 'predictions' in result2:
                        for prediction2 in result2['predictions']:
                            confidence2 = prediction2.get('confidence', 'N/A')
                            if confidence2 > seuil:
                                st.write(f"Confiance de la prédiction Gaël : {confidence2}, Numéro : {counter}")
                                if 'points' in prediction2:
                                    points = prediction2.get('points')
                                    for point in points:
                                        x = point['x']
                                        y = point['y']
                                        liste2.append((x, y))
                                    ordered_points2 = sort_points_for_polygon(liste2)
                                    draw = ImageDraw.Draw(new_image)
                                    draw.polygon(ordered_points2, None, "black")
                                    draw.text((prediction2['points'][0]['x'], prediction2['points'][0]['y']), str(counter), fill="black", font=font)
                                    counter += 1
                    else:
                        st.write("Aucune prédiction trouvée.")
                except Exception as e:
                    st.write(f"Erreur lors de l'inférence block 2 : {e}")

                try:
                    font_path = "sections/nailsdetection/font/Roboto-Bold.ttf"
                    font = ImageFont.truetype(font_path, size=30)
                    result3 = result_non_traite
                    liste3 = []
                    if 'predictions' in result3:
                        for prediction3 in result3['predictions']:
                            confidence3 = prediction3.get('confidence', 'N/A')
                            if confidence3 > seuil:
                                st.write(f"Confiance de la prédiction Gaël, image non traitée : {confidence3}, Numéro : {counter}")
                                if 'points' in prediction3:
                                    points = prediction3.get('points')
                                    for point in points:
                                        x = point['x']
                                        y = point['y']
                                        liste3.append((x, y))
                                    ordered_points3 = sort_points_for_polygon(liste3)
                                    draw = ImageDraw.Draw(ima)
                                    draw.polygon(ordered_points3, None, "black")
                                    draw.text((prediction3['points'][0]['x'], prediction3['points'][0]['y']), str(counter), fill="black", font=font)
                                    counter += 1
                    else:
                        st.write("Aucune prédiction trouvée.")
                except Exception as e:
                    st.write(f"Erreur lors de l'inférence block 3 : {e}")

                try:
                    font_path = "sections/nailsdetection/font/Roboto-Bold.ttf"
                    font = ImageFont.truetype(font_path, size=30)
                    result = result
                    liste = []
                    if 'predictions' in result:
                        for prediction in result['predictions']:
                            confidence = prediction.get('confidence', 'N/A')
                            st.write(f"Confiance de la prédiction Fabrice : {confidence}, Numéro {counter} ")
                            if confidence > seuil:
                                if 'points' in prediction:
                                    points = prediction.get('points')
                                    for point in points:
                                        x = point['x']
                                        y = point['y']
                                        liste.append((x, y))
                                    ordered_points = sort_points_for_polygon(liste)
                                    draw = ImageDraw.Draw(new_image)
                                    draw.polygon(ordered_points, None, "white")
                                    draw.text((prediction['points'][0]['x'], prediction['points'][0]['y']), str(counter), fill="white", font=font)
                                    counter += 1
                    else:
                        st.write("Aucune prédiction trouvée.")
                except Exception as e:
                    st.write(f"Erreur lors de l'inférence block 4 : {e}")

                try:
                    font_path = "sections/nailsdetection/font/Roboto-Bold.ttf"
                    font = ImageFont.truetype(font_path, size=30)
                    result4 = result_non_traite
                    liste4 = []
                    if 'predictions' in result4:
                        for prediction4 in result4['predictions']:
                            confidence4 = prediction4.get('confidence', 'N/A')
                            if confidence4 > seuil:
                                st.write(f"Confiance de la prédiction Fabrice, Image non traitée : {confidence4}, Numéro {counter} ")
                                if 'points' in prediction4:
                                    points = prediction4.get('points')
                                    for point in points:
                                        x = point['x']
                                        y = point['y']
                                        liste4.append((x, y))
                                    ordered_points4 = sort_points_for_polygon(liste4)
                                    draw = ImageDraw.Draw(ima)
                                    draw.polygon(ordered_points4, None, "white")
                                    draw.text((prediction4['points'][0]['x'], prediction4['points'][0]['y']), str(counter), fill="white", font=font)
                                    counter += 1
                    else:
                        st.write("Aucune prédiction trouvée.")
                except Exception as e:
                    st.write(f"Erreur lors de l'inférence block 5 : {e}")

            else:
                counter = 1
                font_path = "sections/nailsdetection/font/Roboto-Bold.ttf"
                font = ImageFont.truetype(font_path, size=30)
                try:
                    liste5 = []
                    if 'predictions' in result:
                        for prediction5 in result['predictions']:
                            confidence5 = prediction5.get('confidence', 'N/A')
                            if confidence5 > seuil:
                                if selected_option == "Modèle de GAEL":
                                    st.write(f"Confiance de la prédiction Gaël : {confidence5}, Numéro : {counter}")
                                elif selected_option == "Modèle avancé de Fabrice" or selected_option == "Modèle basique de Fabrice":
                                    st.write(f"Confiance de la prédiction Fabrice : {confidence5}, Numéro : {counter}")
                                elif selected_option == "Nouveau modèle" and input1 and input2:
                                    st.write(f"Confiance de la prédiction Nouveau modèle : {confidence5}, Numéro : {counter}")
                                if 'points' in prediction5:
                                    points = prediction5.get('points')
                                    for point in points:
                                        x = point['x']
                                        y = point['y']
                                        liste5.append((x, y))
                                    ordered_points = sort_points_for_polygon(liste5)
                                    draw = ImageDraw.Draw(new_image)
                                    if selected_option == "Modèle de GAEL":
                                        draw.polygon(ordered_points, None, "black")
                                        draw.text((prediction5['points'][0]['x'], prediction5['points'][0]['y']), str(counter), fill="black", font=font)
                                        counter += 1
                                    elif selected_option == "Modèle avancé de Fabrice" or selected_option == "Modèle basique de Fabrice":
                                        draw.polygon(ordered_points, None, "white")
                                        draw.text((prediction5['points'][0]['x'], prediction5['points'][0]['y']), str(counter), fill="white", font=font)
                                        counter += 1
                                    elif selected_option == "Nouveau modèle" and input1 and input2:
                                        draw.polygon(ordered_points, None, "white")
                                        draw.text((prediction5['points'][0]['x'], prediction5['points'][0]['y']), str(counter), fill="white", font=font)
                                        counter += 1
                    else:
                        st.write("Aucune prédiction trouvée.")
                    liste6 = []
                    if 'predictions' in result_non_traite:
                        for prediction6 in result_non_traite['predictions']:
                            confidence6 = prediction6.get('confidence', 'N/A')
                            if confidence6 > seuil:
                                if selected_option == "Modèle de GAEL":
                                    st.write(f"Confiance de la prédiction Gaël, image non traitée : {confidence6}, Numéro : {counter}")
                                elif selected_option == "Modèle avancé de Fabrice" or selected_option == "Modèle basique de Fabrice":
                                    st.write(f"Confiance de la prédiction Fabrice, image non traitée : {confidence6}, Numéro : {counter}")
                                elif selected_option == "Nouveau modèle" and input1 and input2:
                                    st.write(f"Confiance de la prédiction Nouveau modèle, image non traitée : {confidence6}, Numéro : {counter}")
                                if 'points' in prediction6:
                                    points = prediction6.get('points')
                                    for point in points:
                                        x = point['x']
                                        y = point['y']
                                        liste6.append((x, y))
                                    ordered_points = sort_points_for_polygon(liste6)
                                    draw = ImageDraw.Draw(ima)
                                    if selected_option == "Modèle de GAEL":
                                        draw.polygon(ordered_points, None, "black")
                                        draw.text((prediction6['points'][0]['x'], prediction6['points'][0]['y']), str(counter), fill="black", font=font)
                                        counter += 1
                                    elif selected_option == "Modèle avancé de Fabrice" or selected_option == "Modèle basique de Fabrice":
                                        draw.polygon(ordered_points, None, "white")
                                        draw.text((prediction6['points'][0]['x'], prediction6['points'][0]['y']), str(counter), fill="white", font=font)
                                        counter += 1
                                    elif selected_option == "Nouveau modèle" and input1 and input2:
                                        draw.polygon(ordered_points, None, "white")
                                        draw.text((prediction6['points'][0]['x'], prediction6['points'][0]['y']), str(counter), fill="white", font=font)
                                        counter += 1
                    else:
                        st.write("Aucune prédiction trouvée.")
                except Exception as e:
                    st.write(f"Erreur lors de l'inférence block 6 : {e}")
                    st.write("Si nouveau modèle, entrer les deux inputs de Robotflow pour celui-ci avant de charger l'image")

        with col2:
            st.image(ima, caption='Image téléchargée - non traitée')
            st.image(new_image, caption='Image \"processed\"')