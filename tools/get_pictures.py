'''
Neste trabalho vocês deverão detectar o robô nos vídeos das 4 câmeras do espaço inteligente e obter a reconstrução da sua posição 3D no mundo. Feito isso, vocês deverão gerar um gráfico da posição do robô, 
mostrando a trajetória que ele realizou.

Para detectar o robô será usado um marcador ARUCO acoplado a sua plataforma. Rotinas de detecção desse tipo de marcador poderão ser usadas para obter sua posição central, assim como as suas quinas nas imagens.
Essas informações, juntamente com os dados de calibração das câmeras, poderão ser usadas para localização 3D do robô.

Informações a serem consideradas:

- Só é necessário a reconstrução do ponto central do robô (ou suas quinas, se vocês acharem melhor). Para isso, vocês podem usar o método explicado no artigo fornecido como material 
adicional ou nos slides que discutimos em sala de aula.

- O robô está identificado por um marcador do tipo ARUCO - Código ID 0 (zero) - Tamanho 30 x 30 cm

- Os vídeos estão sincronizados para garantir que, a cada quadro, vocês estarão processando imagens do robô capturadas no mesmo instante.

- A calibração das câmeras é fornecida em 4 arquivos no formato JSON (Junto com os arquivos JSON estou fornecendo uma rotina para leitura e extração dos dados de calibração).

- Rotinas de detecção dos marcadores Aruco em imagens e vídeo são fornecidas para ajudar no desenvolvimento do trabalho.

ATENÇÃO: Existem rotinas de detecção de ARUCO que já fornecem sua localização e orientação 3D, se a calibração da câmera e o tamanho do padrão forem fornecidas. 
Essas rotinas poderão ser usadas para fazer comparações com a reconstrução 3D fornecida pelo trabalho de vocês, mas não serão aceitas como o trabalho a ser feito. 
Portanto, lembrem-se que vocês deverão desenvolver a rotina de reconstrução, a partir da detecção do ARUCO acoplado ao robô nas imagens 2D capturadas nos vídeos.


DATA DE ENTREGA: 17/03/2023
'''

import cv2
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import datetime
import sys
import socket
from typing import Tuple, Dict

from is_msgs.image_pb2 import Image
from is_project.detector import Detector
from google.protobuf.json_format import Parse
from is_project.conf.options_pb2 import ServiceOptions
from is_wire.core import Channel, Message, Subscription
from google.protobuf.message import Message as PbMessage

# Load options from JSON file
with open('options.json', 'r') as f:
    options = json.load(f)

# Get the camera indices to use
camera_indices = options['cameras']
print(f"Using cameras: {camera_indices}")

# StreamChannel class for live camera feed
class StreamChannel(Channel):
    def __init__(
        self, uri: str = "amqp://guest:guest@localhost:5672", exchange: str = "is"
    ) -> None:
        super().__init__(uri=uri, exchange=exchange)

    def consume_last(self) -> Tuple[Message, int]:
        """
        Consume the last available message from the channel.
        """
        dropped = 0
        msg = super().consume()
        while True:
            try:
                # will raise an exception when no message remained
                msg = super().consume(timeout=0.0)
                dropped += 1
            except socket.timeout:
                return (msg, dropped)

# Function to convert Protocol Buffer Image to NumPy array
def to_np(image: Image) -> np.ndarray:
    """
    Convert a Protocol Buffer Image message to a NumPy array.
    """
    buffer = np.frombuffer(image.data, dtype=np.uint8)
    output = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    return output

# Function to load JSON data
def load_json(filename: str, schema: PbMessage) -> PbMessage:
    """
    Load data from a JSON file and parse it into a Protocol Buffer message.
    """
    with open(file=filename, mode="r", encoding="utf-8") as f:
        proto = Parse(f.read(), schema())
    return proto

def reconstruct_3d_position(corners_list, K_list, R_list, T_list):
    """
    Reconstrói a posição 3D do marcador Aruco com base nas projeções em múltiplas câmeras.
    """
    # Debug information
    # print("Corners list:", corners_list)
    
    A = []
    detected_cameras = 0

    for i in range(len(corners_list)):
        if corners_list[i] is None:
            # print(f"Camera {i}: No marker detected")
            continue  # Ignora câmeras que não detectaram o Aruco
        
        detected_cameras += 1
        # Coordenadas do centro do Aruco na imagem (pixels)
        u, v = corners_list[i]
        # print(f"Camera {i}: Marker detected at ({u}, {v})")
        
        # Use a matriz de projeção correta, assumindo que T é a posição da câmera no mundo
        # Note: We should use R_inv and T_inv here for the camera-to-world transformation
        P = K_list[i] @ np.hstack((R_list[i].T, -R_list[i].T @ T_list[i]))
        # print(f"Camera {i} Projection Matrix P shape: {P.shape}")
        
        # Cria as equações lineares para u e v
        eq1 = u * P[2, :] - P[0, :]
        eq2 = v * P[2, :] - P[1, :]
        # print(f"Camera {i} equation 1: {eq1}")
        # print(f"Camera {i} equation 2: {eq2}")
        
        A.append(eq1)
        A.append(eq2)
    
    # print(f"Detected cameras: {detected_cameras}, total equations: {len(A)}")
    if len(A) < 4:
        # print("ERROR: Need at least 2 cameras (4 equations) for triangulation")
        return None  # Precisa de pelo menos 2 câmeras para triangulação (2 eqs/câmera)
    
    A = np.vstack(A)
    # print(f"Matrix A shape: {A.shape}")
    
    # Resolver pelo SVD (última linha de Vt é a solução)
    try:
        _, s, Vt = np.linalg.svd(A)
        # print(f"SVD singular values: {s}")
        X_homog = Vt[-1]
        # print(f"Homogeneous coordinates: {X_homog}")
        
        # Converter coordenadas homogêneas para cartesianas
        X = X_homog[:3] / X_homog[3]
        return X.flatten()
    except Exception as e:
        # print(f"SVD computation error: {e}")
        return None
    
def reconstruct_3d_position_opencv(corners_list, K_list, R_list, T_list):
    """
    Reconstruct the 3D position of the Aruco marker using OpenCV's triangulation.
    """
    # Filter out None values
    valid_corners = []
    valid_projection_matrices = []
    
    for i in range(len(corners_list)):
        if corners_list[i] is None:
            continue  # Skip cameras that didn't detect the marker
            
        # Get image point
        u, v = corners_list[i]
        point_2d = np.array([[u], [v]], dtype=np.float64)
        valid_corners.append(point_2d)
        
        # Compute projection matrix P = K[R|t]
        P = K_list[i] @ np.hstack((R_list[i], T_list[i]))
        valid_projection_matrices.append(P)
    
    # Need at least two cameras for triangulation
    if len(valid_corners) < 2:
        return None
        
    # Convert lists to proper format for triangulatePoints
    points_2d = np.array(valid_corners)
    
    # Use first two cameras for initial triangulation
    point_4d = cv2.triangulatePoints(
        valid_projection_matrices[0], 
        valid_projection_matrices[1],
        points_2d[0], 
        points_2d[1]
    )
    
    # Convert homogeneous coordinates to 3D
    point_3d = point_4d[:3] / point_4d[3]
    
    return point_3d
    
def plot_camera_axes(R, T, ax, scale=0.5, label=''):
    """
    Plota os eixos do referencial da câmera.
    - R: matriz de rotação (3x3), onde cada coluna representa um eixo (x, y, z)
    - T: vetor de translação (3x1), posição da câmera no ambiente
    - ax: objeto Axes3D onde será plotado
    - scale: comprimento das setas para visualização
    - label: rótulo para identificar a câmera
    """
    # Converte T para array 1D
    origin = T.flatten()
    
    # Vetores dos eixos da câmera em coordenadas globais
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]
    
    # Plota os eixos usando ax.quiver
    ax.quiver(origin[0], origin[1], origin[2],
              x_axis[0], x_axis[1], x_axis[2],
              color='r', length=scale, normalize=True)
    ax.quiver(origin[0], origin[1], origin[2],
              y_axis[0], y_axis[1], y_axis[2],
              color='g', length=scale, normalize=True)
    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0], z_axis[1], z_axis[2],
              color='b', length=scale, normalize=True)
    
    # Adiciona um rótulo na posição da câmera
    ax.text(origin[0], origin[1], origin[2], label, fontsize=10)


# Function to read the intrinsic and extrinsic parameters of each camera
def camera_parameters(file):
    camera_data = json.load(open(file))
    K = np.array(camera_data['intrinsic']['doubles']).reshape(3, 3)
    res = [camera_data['resolution']['width'],
           camera_data['resolution']['height']]
    # Fix: Access extrinsic as an array
    tf = np.array(camera_data['extrinsic'][0]['tf']['doubles']).reshape(4, 4)
    
    # Get the RT matrix (original camera-to-world transformation)
    R = tf[:3, :3]
    T = tf[:3, 3].reshape(3, 1)
    
    # Compute the inverse transformation (world-to-camera)
    # For a rotation matrix, its inverse is its transpose
    R_inv = R.transpose()
    # For translation, we need to apply -T rotated by inverse R
    T_inv = -R_inv @ T
    
    dis = np.array(camera_data['distortion']['doubles'])
    return K, R, T, res, dis, R_inv, T_inv

# Load camera parameters for each camera in the options
cameras_data = {}
for cam_idx in camera_indices:
    json_file = f"calibrations/{cam_idx}.json"
    K, R, T, res, dis, R_inv, T_inv = camera_parameters(json_file)
    cameras_data[cam_idx] = {
        'K': K, 'R': R, 'T': T, 'res': res, 'dis': dis, 
        'R_inv': R_inv, 'T_inv': T_inv
    }
    
    # print(f'Camera {cam_idx}\n')
    # print('Resolucao', res, '\n')
    # print('Parametros intrinsecos:\n', K, '\n')
    # print('Parametros extrinsecos:\n')
    # print(f'R{cam_idx}\n', R, '\n')
    # print(f'T{cam_idx}\n', T, '\n')
    # print('Distorcao Radial:\n', dis)

parameters = aruco.DetectorParameters()
dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoDetector = aruco.ArucoDetector(dictionary, parameters)

# Initialize camera connections
options_service = load_json(
    filename="options.json",
    schema=ServiceOptions,
)

# Create dictionaries to store channels and subscriptions for each camera
channels = {}
subscriptions = {}

# Create a channel and subscription for each camera
for cam_idx in camera_indices:
    channels[cam_idx] = StreamChannel(f"amqp://guest:guest@{options['address']}")
    subscriptions[cam_idx] = Subscription(channels[cam_idx], name=f"PeopleDetector{cam_idx}")
    subscriptions[cam_idx].subscribe(topic=f"CameraGateway.{cam_idx}.Frame")
    print(f"Subscribed to camera {cam_idx}")

def create_mosaic(frames, camera_indices, target_width=1920, target_height=1080):
    """
    Create a mosaic of frames arranged in a grid layout.
    
    Parameters:
    -----------
    frames : dict
        Dictionary with camera indices as keys and frames as values
    camera_indices : list
        List of camera indices to arrange
    target_width : int
        Target width for the output mosaic
    target_height : int
        Target height for the output mosaic
        
    Returns:
    --------
    numpy.ndarray
        Mosaic of frames arranged in a grid
    """
    # Filter out None frames
    valid_frames = {idx: frames[idx] for idx in camera_indices if idx in frames and frames[idx] is not None}
    
    if not valid_frames:
        return None
    
    # Determine grid dimensions based on number of cameras
    n_frames = len(valid_frames)
    grid_cols = min(2, n_frames)  # Max 2 columns
    grid_rows = (n_frames + grid_cols - 1) // grid_cols  # Ceiling division
    
    # Calculate frame size to fit in the grid
    frame_width = target_width // grid_cols
    frame_height = target_height // grid_rows
    
    # Create the grid
    mosaic = np.zeros((grid_rows * frame_height, grid_cols * frame_width, 3), dtype=np.uint8)
    
    # Place frames in the grid
    for i, cam_idx in enumerate(sorted(valid_frames.keys())):
        frame = valid_frames[cam_idx]
        # Calculate position in grid
        row = i // grid_cols
        col = i % grid_cols
        
        # Resize frame to fit in the grid
        resized_frame = cv2.resize(frame, (frame_width, frame_height))
        
        # Add camera index label
        cv2.putText(resized_frame, f"Camera {cam_idx}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Place in mosaic
        y_start = row * frame_height
        y_end = y_start + frame_height
        x_start = col * frame_width
        x_end = x_start + frame_width
        
        mosaic[y_start:y_end, x_start:x_end] = resized_frame
    
    return mosaic

def save_picture_and_position(frames, X, camera_indices, output_dir="captures"):
    """
    Save pictures from all cameras and the marker position to a JSON file.
    
    Parameters:
    -----------
    frames : dict
        Dictionary with camera indices as keys and frames as values
    X : numpy.ndarray
        3D position of the marker
    camera_indices : list
        List of camera indices
    output_dir : str
        Directory to save the captures
    
    Returns:
    --------
    str
        Path to the saved data directory
    """
    # Create a more descriptive folder name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_name = f"marker_position_{timestamp}"
    
    # Create output directory if it doesn't exist
    capture_dir = os.path.join(output_dir, capture_name)
    os.makedirs(capture_dir, exist_ok=True)
    
    # Save frames from all cameras with more descriptive filenames
    saved_images = {}
    for cam_idx in camera_indices:
        if cam_idx in frames and frames[cam_idx] is not None:
            # Create a more descriptive filename
            img_filename = f"aruco_marker7_camera{cam_idx}.jpg"
            img_path = os.path.join(capture_dir, img_filename)
            cv2.imwrite(img_path, frames[cam_idx])
            saved_images[cam_idx] = img_filename  # Store just the filename, not full path
            
    # Create JSON with position data and more metadata
    position_data = {
        "capture_timestamp": timestamp,
        "marker_id": 7,
        "marker_position_3d": {
            "x": float(X[0][0]),
            "y": float(X[0][1]),
            "z": float(X[0][2])
        },
        "camera_images": saved_images,
        "camera_indices_used": camera_indices
    }
    
    # Save to JSON file with a more descriptive name
    json_path = os.path.join(capture_dir, "marker_position_data.json")
    with open(json_path, 'w') as f:
        json.dump(position_data, f, indent=4)
        
    print(f"Saved capture to {capture_dir}")
    return capture_dir

#####################################PLOT###############################################
# Cria figura e eixo 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plota o ponto (0,0,0) em preto (origem do ambiente)
ax.scatter(0, 0, 0, color='k', s=50, marker='x')

# Plot das cameras no espaço
for cam_idx in camera_indices:
    plot_camera_axes(
        cameras_data[cam_idx]['R_inv'], 
        cameras_data[cam_idx]['T_inv'], 
        ax, scale=0.5, 
        label=f'Câmera {cam_idx}'
    )

trajetoria, = ax.plot([], [], [], marker='o', linestyle='', color='r', label='Posições do Marcador')

# Configurações adicionais do gráfico
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Posições 3D do Marcador')

# Criar uma nova figura para o plot 2D
fig2d = plt.figure()
ax2d = fig2d.add_subplot(111)
trajetoria2d, = ax2d.plot([], [], marker='o', linestyle='', color='b', label='Posições do Marcador')
ax2d.set_xlabel('X')
ax2d.set_ylabel('Y')
ax2d.set_title('Posição do Robô no Plano X-Y')
ax2d.set_xlim((-2, 2))
ax2d.set_ylim((-2, 2))
ax2d.grid(True)
ax2d.legend()

plt.ion()
plt.pause(0.05)
X_hist = np.empty((0, 3))
#####################################################################################

# Create output directory for captures
capture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
os.makedirs(capture_dir, exist_ok=True)

while True:
    # Read frames from all cameras
    frames = {}
    frame_empty = True
    
    # Consume the latest message from each camera
    for cam_idx in camera_indices:
        try:
            message, _ = channels[cam_idx].consume_last()
            image = message.unpack(Image)
            frame = to_np(image)
            frames[cam_idx] = frame
            if frame is not None:
                frame_empty = False
        except Exception as e:
            # frames[cam_idx] = None
    
            if frame_empty:
                # Don't break - just keep trying for live feed
                plt.pause(0.1)
                continue
    
    # Detect markers in all frames
    corners_dict = {}
    ids_dict = {}
    marked_frames = {}
    
    for cam_idx in camera_indices:
        if frames[cam_idx] is None:
            continue
            
        corners, ids, _ = arucoDetector.detectMarkers(frames[cam_idx])
        corners_dict[cam_idx] = corners
        ids_dict[cam_idx] = ids
        
        # Draw markers on frames
        marked_frames[cam_idx] = aruco.drawDetectedMarkers(frames[cam_idx].copy(), corners, ids)
    
    # Get center points of marker 7 for each camera
    corners_list = [None] * len(camera_indices)
    
    for i, cam_idx in enumerate(camera_indices):
        if camera_indices[i] not in corners_dict or corners_dict[camera_indices[i]] is None or ids_dict[camera_indices[i]] is None:
            continue
        
        corners = corners_dict[camera_indices[i]]
        ids = ids_dict[camera_indices[i]]
        
        if ids is not None and 7 in ids:
            idx = np.where(ids == 7)[0][0]
            c = corners[idx][0]
            cx, cy = np.mean(c, axis=0)
            corners_list[i] = (cx, cy)
    
    # Prepare data for 3D reconstruction
    K_list = [cameras_data[cam_idx]['K'] for cam_idx in camera_indices]
    R_list = [cameras_data[cam_idx]['R_inv'] for cam_idx in camera_indices]
    T_list = [cameras_data[cam_idx]['T_inv'] for cam_idx in camera_indices]
    
    # Try reconstruction
    X = reconstruct_3d_position(corners_list, K_list, R_list, T_list)
    
    if X is not None:
        X = np.array(X).reshape(1, 3)
        X_hist = np.vstack((X_hist, X))
        
        trajetoria.set_data(X_hist[:, 0], X_hist[:, 1])
        trajetoria.set_3d_properties(X_hist[:, 2])
        
        # Atualizar o plot 2D apenas com pontos
        trajetoria2d.set_data(X_hist[:, 0], X_hist[:, 1])
        fig2d.canvas.draw_idle()
    else:
        pass
    
    plt.draw()
    plt.pause(0.005)
    
    # Display frames
    if len(marked_frames) > 0:
        # Create mosaic of all frames with detected markers
        mosaic = create_mosaic(marked_frames, camera_indices)
        if mosaic is not None:
            cv2.imshow('Detected Markers', mosaic)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('k') and X is not None:
        # Save the current frame and position when 'k' is pressed
        save_picture_and_position(
            marked_frames, 
            X, 
            camera_indices, 
            output_dir=capture_dir
        )
        print(f"Captured frames and position: {X}")

# No need to release video captures for live feed
cv2.destroyAllWindows()