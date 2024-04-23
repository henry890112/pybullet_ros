import socket
import numpy as np
import ast

# TCP 傳送
def send_matrix(host, port, matrix):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        matrix_str = str(matrix.tolist())  # 將矩陣轉換為字串
        s.sendall(matrix_str.encode())

# TCP 接收 (接收端必須先啟動等待連線並傳送)
def receive_matrix(connection):
    received_data = connection.recv(1024).decode()
    matrix = ast.literal_eval(received_data)  # 將字串轉換回矩陣
    return np.array(matrix)

def start_server(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print("伺服器啟動，等待連接並接收...")
        conn, addr = s.accept()
        with conn:
            print(f"來自 {addr} 的連接已建立")
            matrix = receive_matrix(conn)
            print("接收到的矩陣：")
            print(matrix)
            return matrix

def start_server_checker(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"服務端啟動，等待連接於 {host}:{port}...")

        conn, addr = server_socket.accept()
        with conn:
            print(f"來自 {addr} 的連接已建立。")

            while True:
                data = conn.recv(1024).decode()
                if not data:
                    break  # 如果客戶端關閉連接，跳出循環
                print(f"接收到數據：{data}")
                if data == "checker":
                    print("接收到checker，服務端將繼續執行。")
                    # 在這裡執行後續的操作
                    return "checker"  # 返回接收到的checker訊息

def start_server_target_name(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print(f"服務端正在監聽 {host}:{port}...")

        conn, addr = server_socket.accept()
        with conn:
            print(f"從 {addr} 建立了連接")
            while True:
                data = conn.recv(1024)
                if not data:
                    break  # 客戶端關閉連接
                received_message = data.decode()
                print(f"接收到的訊息：{received_message}")
                return received_message

# TCP 傳送
        
def send_checker(host, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    message = "checker"
    client_socket.sendall(message.encode())
    print("已發送checker到服務端。")
    client_socket.close()

def send_target_name(host, port, message):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((host, port))
        client_socket.sendall(message.encode())  # 直接傳送字串
        print(f"已向服務端發送：{message}")
