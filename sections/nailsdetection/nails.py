import streamlit as st
from PIL import Image, ImageOps, ImageDraw
from inference_sdk import InferenceHTTPClient
import pandas as pd


def draw_polygon_pil(points, image_size=(300, 300), fill_color="black", outline_color="black"):
    """
    Dessine un polygone sur une image à l'aide de PIL.

    Paramètres :
        points (list): Liste de tuples (x, y) représentant les sommets du polygone.
        image_size (tuple): Taille de l'image (largeur, hauteur).
        fill_color (str): Couleur de remplissage du polygone.
        outline_color (str): Couleur du contour du polygone.
    """
    # Crée une image blanche
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)
    
    # Dessine le polygone
    draw.polygon(points, fill=fill_color, outline=outline_color)
    
    # Affiche l'image
    image.show()


import math
# Calculer la distance entre deux points
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
# Trier les points pour former un polygone
def sort_points_for_polygon(points):
    if not points:
        return []
    # Initialiser les variables
    ordered_points = []
    current_point = points.pop(0)  # Prendre un point arbitraire comme point de départ
    ordered_points.append(current_point)
    # Répéter jusqu'à ce qu'il ne reste plus de points
    while points:
        # Trouver le point le plus proche du point actuel
        closest_point = min(points, key=lambda p: distance(current_point, p))
        ordered_points.append(closest_point)
        points.remove(closest_point)
        current_point = closest_point
    return ordered_points

def nail_page():
    st.header("Bienvenue")
    st.caption("Bienvenue dans la détection d'ongle")
    #container = st.container(border=True)
    
    c1, _ = st.columns([4,2], vertical_alignment="center")
    
    # with c1 : Raccourcir les barres de choix du modèle puis de l'image
    with c1:
        # Choix des modèles
        options = ["Modèle de GAEL", "Modèle basique de Fabrice", "Modèle avancé de Fabrice", "Superposer les modèles", "Nouveau modèle"]
        
        # Créer une liste déroulante
        selected_option = st.selectbox("Choisissez un modèle :", options)
        if selected_option == "Nouveau modèle": 
            # Champ de saisie 1
            input1 = st.text_input("Entrez l'api_key de Robotflow :", key="premier bouton")
            # Champ de saisie 2
            input2 = st.text_input("Entrez le model_id de Robotflow :", key = "second bouton")
        
        # Création d'un bouton radio pour confiance de la prédiction
        choice = st.radio(
            "Choisissez un seuil en pourcentage pour la confiance de prédiction :",
            ("50%", "60%", "70%", "80%")
        )

        # Logique basée sur le choix
        if choice == "50%":
            seuil = 0.5
        elif choice == "60%":
            seuil = 0.6
        elif choice =="70%":
            seuil = 0.7
        elif choice == "80%":
            seuil = 0.8
        
        
        # Créez un uploader pour les fichiers
        uploaded_file = st.file_uploader("Choisissez une image", type=["png", "jpg", "jpeg"])
    
    col1, col2 = st.columns(2, vertical_alignment="center")
    
    # Vérifiez si un fichier a été téléchargé
    if uploaded_file is not None:
        # Ouvrir l'image avec PIL
        ima = Image.open(uploaded_file)
        image = ima
            #im1 = processed_image_path
            # save a image using extension
        im_non_traitee = image.save("geeks.jpg")
            
        with col1 :
        
            ## TRAITEMENT GAEL   
            # Orientation automatique de l'image
            image = ImageOps.exif_transpose(image)

            # Taille cible de l'image
            target_size = (340, 340)

            # Calcule le ratio de redimensionnement
            ratio = min(target_size[0] / image.width, target_size[1] / image.height)

            # Calcule la nouvelle taille de l'image
            new_size = (int(image.width * ratio), int(image.height * ratio))

            # Redimensionne l'image
            image = image.resize(new_size, Image.LANCZOS)

            # Crée une nouvelle image noire de la taille cible
            new_image = Image.new('L', target_size, 'black')

            # Calcule la position pour centrer l'image redimensionnée dans le nouveau fond noir
            left = (target_size[0] - new_size[0]) // 2
            top = (target_size[1] - new_size[1]) // 2

            # cole la photo redimensionnée dans le nouveau fond noir
            new_image.paste(image, (left, top))

            # Enregistre et traite l'image
            processed_image_path = 'IMG_2075_processed.jpg'
            new_image.save(processed_image_path)

            # Verification de la taille de sortie de l'image
            print(f"format de sortie: {new_image.size}")

            # Conversion en noir et blanc grayscale
            new_image = new_image.convert("L")

            # Sauvegarde de l'image traitée
            processed_image_path = "IMG_2075_processed.jpg"
            new_image.save(processed_image_path)

            try: 
                # initialiser le client robotflow
                if selected_option == "Modèle de GAEL":
                    GAEL = InferenceHTTPClient(
                                api_url="https://detect.roboflow.com",
                                api_key="C1QXXjGrgpWRBq8uQfS7"
                            )
                    result = GAEL.infer(processed_image_path, model_id="nails-diginamic-7syht/1")
                else:
                    if selected_option == "Modèle basique de Fabrice":
                        CLIENT = InferenceHTTPClient(
                                api_url="https://detect.roboflow.com",
                                api_key="D7dNUce8UyrrxqFcplzH"
                            )
                        result = CLIENT.infer(processed_image_path, model_id="nails-diginamic-el24e/2")
                    else:
                        if selected_option == "Modèle avancé de Fabrice":
                            CLIENT = InferenceHTTPClient(
                                api_url="https://detect.roboflow.com",
                                api_key="D7dNUce8UyrrxqFcplzH"
                            )
                            result = CLIENT.infer(processed_image_path, model_id="nails-diginamic-el24e/3")
                        else:
                            if selected_option == "Nouveau modèle" and input1 and input2:
                                CLIENT = InferenceHTTPClient(
                                api_url="https://detect.roboflow.com",
                                api_key= input1
                                )
                                result = CLIENT.infer(processed_image_path, model_id= input2)
            except Exception as e:
                st.write(f"Erreur lors de l'inférence block 1 : {e}, si Nouveau modèle, entrer les deux valeurs d'input")
                
            if selected_option == "Superposer les modèles":    
                # Perform inference / Image traitée
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
                            if confidence2 > seuil : #0.5:
                                centerX= prediction2['x']
                                centerY= prediction2['y']
                                st.write(f"Confiance de la prédiction Gaël : {confidence2}")
                                if 'points' in prediction2:  # Vérifiez la présence de l'annotation polygonale
                                    points = prediction2.get('points')
                                    for point in points:
                                        x = point['x']
                                        y = point['y']
                                        liste2.append((x,y))
                                            
                                    ordered_points2 = sort_points_for_polygon(liste2)
                                    draw = ImageDraw.Draw(new_image)
                                    # Dessine le polygone
                                    draw.polygon(ordered_points2, None, "green")
                                    
                                    #st.write(liste2)
                                            
                                    # Affiche l'image
                                    #new_image.show()
                                    #draw_polygon_pil(liste,(300,300),"black","black")
                                else:
                                    print("Aucune annotation de polygone trouvée.")
                    else:
                        st.write("Aucune prédiction trouvée.")
                except Exception as e:
                    st.write(f"Erreur lors de l'inférence block 2 : {e}")
                
                #Image non traitée
                try:
                    GAEL = InferenceHTTPClient(
                        api_url="https://detect.roboflow.com",
                        api_key="C1QXXjGrgpWRBq8uQfS7"
                    )
                    result3 = GAEL.infer("geeks.jpg", model_id="nails-diginamic-7syht/1")
                    
                    if 'predictions' in result3:
                        for prediction3 in result3['predictions']:
                            confidence3 = prediction3.get('confidence', 'N/A')
                            if confidence3 > seuil: # 0.5 : 
                                st.write(f"Confiance de la prédiction Gaël, image non traitée : {confidence3}")
                            
                    else:
                        st.write("Aucune prédiction trouvée.")
                except Exception as e:
                    st.write(f"Erreur lors de l'inférence block 3 : {e}")
                
                # Perform inference : Image traitée, autre robotflow
                try:
                    CLIENT = InferenceHTTPClient(
                        api_url="https://detect.roboflow.com",
                        #api_key="D7dNUce8UyrrxqFcplzH"
                        api_key="D7dNUce8UyrrxqFcplzH"
                    )
                    result = CLIENT.infer(processed_image_path, model_id="nails-diginamic-el24e/3") #nails-diginamic-el24e/2")
                    liste = []
                    if 'predictions' in result:
                        for prediction in result['predictions']:
                            confidence = prediction.get('confidence', 'N/A')
                            st.write(f"Confiance de la prédiction Fabrice : {confidence}")
                            if confidence > seuil : #0.5:
                                if 'points' in prediction:  # Vérifiez la présence de l'annotation polygonale
                                    points = prediction.get('points')
                                    for point in points:
                                        x = point['x']
                                        y = point['y']
                                        liste.append((x,y))
                                            
                                    ordered_points = sort_points_for_polygon(liste)
                                    draw = ImageDraw.Draw(new_image)
                                    # Dessine le polygone
                                    draw.polygon(ordered_points, None, "white")
                                else:
                                    print("Aucune annotation de polygone trouvée.")
                    else:
                        st.write("Aucune prédiction trouvée.")
                except Exception as e:
                    st.write(f"Erreur lors de l'inférence block 4 : {e}")
                        
                # Perform inference : Image non traitée, autre robotflow
                try:
                    CLIENT = InferenceHTTPClient(
                        api_url="https://detect.roboflow.com",
                        #api_key="D7dNUce8UyrrxqFcplzH"
                        api_key="D7dNUce8UyrrxqFcplzH"
                    )
                    result = CLIENT.infer("geeks.jpg", model_id="nails-diginamic-el24e/3")
                    #st.write(result)
                    #draw = ImageDraw.Draw(new_image)
                    #liste = []
                    if 'predictions' in result:
                        for prediction in result['predictions']:
                            confidence = prediction.get('confidence', 'N/A')
                            if confidence > seuil : #0.5:
                                st.write(f"Confiance de la prédiction Fabrice, Image non traitée : {confidence}")
                        #     if 'points' in prediction:  # Vérifiez la présence de l'annotation polygonale
                        #         points = prediction.get('points')
                        #         for point in points:
                        #             x = point['x']
                        #             y = point['y']
                        #             liste.append((x,y))
                        #         #    st.write(f"Coordonnées du point : X={x}, Y={y}")
                        #         # Crée le polygone et l'ajoute au tracé
                        #         #polygon = Polygon(points, closed=True, fill=True, edgecolor='blue', facecolor='lightblue', alpha=0.5)
                        #         #draw.add_patch(polygon)
                        # #draw.polygon(points,"black","black", 240)
                        # #draw_polygon_pil(points,new_image,"black","black")
                        #         #st.image(new_image, caption='Image \"processed\"', width=240)
                        # #new_image.show()
                                
                        #         draw = ImageDraw.Draw(image)
                        #         # Dessine le polygone
                        #         draw.polygon(liste, "black", "black")
                                
                        #         # Affiche l'image
                        #         #new_image.show()
                        #         #draw_polygon_pil(liste,(300,300),"black","black")
                        #     else:
                        #         print("Aucune annotation de polygone trouvée.")
                    else:
                        st.write("Aucune prédiction trouvée.")
                except Exception as e:
                    st.write(f"Erreur lors de l'inférence block 5 : {e}")
            
            # Nouveau modèle
            else:
                try:
                    liste = []
                    if 'predictions' in result:
                        for prediction in result['predictions']:
                            confidence = prediction.get('confidence', 'N/A')
                            if confidence > seuil : #0.5:
                                if selected_option == "Modèle de GAEL":
                                    st.write(f"Confiance de la prédiction Gaël : {confidence}")
                                else:
                                    if selected_option == "Modèle avancé de Fabrice" or selected_option == "Modèle basique de Fabrice":
                                            st.write(f"Confiance de la prédiction Fabrice : {confidence}")
                                    else:
                                        if selected_option == "Nouveau modèle" :
                                            if input1 and input2 :
                                                st.write(f"Confiance de la prédiction Nouveau modèle : {confidence}")
                                            else :
                                                st.write("Si nouveau modèle, entrer les deux inputs de Robotflow pour celui-ci avant de charger l'image")
                                if 'points' in prediction:  # Vérifiez la présence de l'annotation polygonale
                                    points = prediction.get('points')
                                    for point in points:
                                        x = point['x']
                                        y = point['y']
                                        liste.append((x,y))
                                    ordered_points = sort_points_for_polygon(liste)
                                    draw = ImageDraw.Draw(new_image)
                                    # Dessine le polygone
                                    if selected_option == "Modèle de GAEL":
                                        draw.polygon(ordered_points, None, "black")
                                    else:
                                        if selected_option == "Modèle avancé de Fabrice" or selected_option == "Modèle basique de Fabrice":
                                            draw.polygon(ordered_points, None, "white")
                                        else:
                                            if selected_option == "Nouveau modèle":
                                                if input1 and input2:
                                                    draw.polygon(ordered_points, None, "gray")
                                                else :
                                                    st.write("Si nouveau modèle, entrer les deux inputs de Robotflow pour celui-ci avant de charger l'image")
                                else:
                                    print("Aucune annotation de polygone trouvée.")
                    else:
                        st.write("Aucune prédiction trouvée.")
                        
                except Exception as e:
                    st.write(f"Erreur lors de l'inférence block 6 : {e}")
                    st.write("Si nouveau modèle, entrer les deux inputs de Robotflow pour celui-ci avant de charger l'image")
                    
        # for predict in result['predictions']:        
        #     st.write(f"Confiance : {predict.get('confidence')}")
        
        with col2 :
            # Afficher l'image
            st.image(image, caption='Image téléchargée')
            st.image(new_image, caption='Image \"processed\"')
            
            

 

 
# # Exemple d'utilisation
# points = [(0, 0), (1, 2), (3, 4), (6, 1), (2, 2)]
# ordered_points = sort_points_for_polygon(points)
# print("Ordre des points :", ordered_points)
 