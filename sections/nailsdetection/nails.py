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


def nail_page():
    st.header("Bienvenue")
    st.caption("Bienvenue dans la détection d'ongle")
    #container = st.container(border=True)
    
    c1, _ = st.columns([4,2], vertical_alignment="center")
    
    with c1:
        # Créez un uploader pour les fichiers
        uploaded_file = st.file_uploader("Choisissez une image", type=["png", "jpg", "jpeg"])
    
    col1, col2 = st.columns(2, vertical_alignment="center")
    
    # initialize the client robotflow
    
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
            target_size = (640, 640)

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

            
            
            ## TRAITEMENT GAEL
            
            # creating a image object (main image)  
            #im1 = image
            #im1 = processed_image_path
            # save a image using extension
            #im1 = im1.save("geeks.jpg")
            # infer on a local image
            #result = CLIENT.infer("/Users/fabricemazenc/Downloads/20241114_101437.JPG", model_id="nails-diginamic-el24e/2") # result 0.745303
            #result = CLIENT.infer(processed_image_path, model_id="nails-diginamic-el24e/2")  #result 0.751093
            
            # Perform inference / Image traitée
            try:
                GAEL = InferenceHTTPClient(
                    api_url="https://detect.roboflow.com",
                    api_key="C1QXXjGrgpWRBq8uQfS7"
                )
                result2 = GAEL.infer(processed_image_path, model_id="nails-diginamic-7syht/1")
                if 'predictions' in result2:
                    for prediction2 in result2['predictions']:
                        confidence2 = prediction2.get('confidence', 'N/A')
                        st.write(f"Confiance de la prédiction Gaël : {confidence2}")
                else:
                    st.write("Aucune prédiction trouvée.")
            except Exception as e:
                st.write(f"Erreur lors de l'inférence : {e}")
                
            #Image non traitée
            try:
                GAEL = InferenceHTTPClient(
                    api_url="https://detect.roboflow.com",
                    api_key="C1QXXjGrgpWRBq8uQfS7"
                )
                result3 = GAEL.infer("geeks.jpg", model_id="nails-diginamic-7syht/1")
                liste = []
                if 'predictions' in result3:
                    for prediction3 in result3['predictions']:
                        confidence3 = prediction3.get('confidence', 'N/A')
                        st.write(f"Confiance de la prédiction Gaël, image non traitée : {confidence3}")
                        if 'points' in prediction3:  # Vérifiez la présence de l'annotation polygonale
                            points = prediction3.get('points')
                            for point in points:
                                x = point['x']
                                y = point['y']
                                liste.append((x,y))
                                
                            draw = ImageDraw.Draw(image)
                            # Dessine le polygone
                            draw.polygon(liste, None, "black")
                            st.write(liste)
                            # Affiche l'image
                            #new_image.show()
                            #draw_polygon_pil(liste,(300,300),"black","black")
                        else:
                            print("Aucune annotation de polygone trouvée.")
                else:
                    st.write("Aucune prédiction trouvée.")
            except Exception as e:
                st.write(f"Erreur lors de l'inférence : {e}")
            
            # Perform inference : Image traitée, autre robotflow
            try:
                CLIENT = InferenceHTTPClient(
                    api_url="https://detect.roboflow.com",
                    api_key="D7dNUce8UyrrxqFcplzH"
                )
                result = CLIENT.infer(processed_image_path, model_id="nails-diginamic-el24e/2")
                if 'predictions' in result:
                    for prediction in result['predictions']:
                        confidence = prediction.get('confidence', 'N/A')
                        st.write(f"Confiance de la prédiction Fabrice : {confidence}")
                else:
                    st.write("Aucune prédiction trouvée.")
            except Exception as e:
                st.write(f"Erreur lors de l'inférence : {e}")
                    
            # Perform inference : Image non traitée, autre robotflow
            try:
                CLIENT = InferenceHTTPClient(
                    api_url="https://detect.roboflow.com",
                    api_key="D7dNUce8UyrrxqFcplzH"
                )
                result = CLIENT.infer("geeks.jpg", model_id="nails-diginamic-el24e/2")
                #st.write(result)
                #draw = ImageDraw.Draw(new_image)
                #liste = []
                if 'predictions' in result:
                    for prediction in result['predictions']:
                        confidence = prediction.get('confidence', 'N/A')
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
                st.write(f"Erreur lors de l'inférence : {e}")
        
                    
        # for predict in result['predictions']:        
        #     st.write(f"Confiance : {predict.get('confidence')}")
        
        with col2 :
            # Afficher l'image
            st.image(image, caption='Image téléchargée')
            st.image(new_image, caption='Image \"processed\"')
            