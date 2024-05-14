"""
Autor: Thiago Garcia João
SoS Rio Grande do Sul
Machine Learning & AI Scientist ESP

""" 



import torch
import torchvision.transforms as T
from torchvision.models.detection import retinanet_resnet50_fpn
import cv2

def main_novo(video_path, output_path=None, analyze_percent=100, frames_to_skip=0):
    # Carregar o modelo RetinaNet pré-treinado
    model = retinanet_resnet50_fpn(pretrained=True)
    model.eval()  # Colocar o modelo em modo de avaliação

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

    # Definir transformações de imagem
    transform = T.Compose([T.ToTensor()])

    # Calcular o número de frames para processar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = int((analyze_percent / 100) * total_frames)

    # Processar cada quadro do vídeo
    frame_count = 0
    while frame_count < frames_to_process:

        print(f"{frame_count}/{frames_to_process}")
        
        ret, frame = cap.read()
        if not ret:
            break

        # Preparar quadro para modelo
        frame_tensor = transform(frame).unsqueeze(0)  # Adicionar dimensão de batch

        # Executar detecção de objetos
        with torch.no_grad():
            results = model(frame_tensor)

        # Desenhar caixas delimitadoras e rótulos nas detecções
        scores = results[0]['scores']
        boxes = results[0]['boxes']
        labels = results[0]['labels']
        for score, box, label in zip(scores, boxes, labels):
            if score > 0.2 and label == 1:  # Filtrar detecções por uma pontuação de confiança e somente pessoas
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'Pessoa {score:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Exibir quadro processado
        #cv2.imshow('Video', frame)

        # Salvar quadro no arquivo de saída, se fornecido
        if output_path:
            out.write(frame)

        # Parar se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Pular frames
        frame_count += 1 + frames_to_skip
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    # Liberar recursos
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Caminho do vídeo de entrada
    video_path = 'DJI_0116.MP4'
    # Caminho para o arquivo de saída (opcional)
    output_path = f'saida_{video_path.replace('.mp4', '')}_resnet50_fpn_2.mp4'
    
    # Quantidade de frames a pular após cada leitura
    frames_to_skip = 1

    # Chamar a função principal
    main_novo(video_path, output_path, analyze_percent=100, frames_to_skip=frames_to_skip)  # Analisar apenas X% do vídeo
