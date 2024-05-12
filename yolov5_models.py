"""
Autor: Thiago Garcia João
SOS RS - IA Voluntários
"""

import torch
import cv2

def main_novo(video_path, output_path = None, model_name = 'yolov5l'):
    # Carregar o modelo YOLOv5-NAS-L
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

    # Abrir o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Configurar saída de vídeo se um caminho de saída for fornecido
    if output_path:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Processar cada quadro do vídeo
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Executar detecção de objetos
        results = model(frame)

        # Desenhar caixas delimitadoras e rótulos nas detecções de pessoas
        for box in results.xyxy[0]:
            x1, y1, x2, y2, conf, class_id = box.tolist()
            class_name = model.names[int(class_id)]
            color = (0, 255, 0)  # Verde para pessoas, azul para outros objetos
            if class_name != 'person':
                color = (255, 0, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f'{class_name} {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Exibir quadro processado
        #cv2.imshow('Video', frame)

        # Salvar quadro no arquivo de saída, se fornecido
        if output_path:
            out.write(frame)

        # Parar se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    # Caminho do vídeo de entrada
    video_path = 'teste_alagamento.mp4'
     # Modelo a ser utilizado 'custom', 'yolov5l', 'yolov5l6', 'yolov5m', 'yolov5m6', 'yolov5n', 'yolov5n6', 'yolov5s', 'yolov5s6', 'yolov5x', 'yolov5x6'
    model_name = 'yolov5x6'
    # Caminho para o arquivo de saída (opcional)
    output_path = f'saida_{model_name}.mp4'

    # Chamar a função principal
    main_novo(video_path, output_path, model_name)
