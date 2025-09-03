import socket


# 서버 주소와 포트
SERVER_HOST = '172.17.16.5'  # 서버 주소
SERVER_PORT = 7800

# 소켓 생성
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 서버에 연결
client_socket.connect((SERVER_HOST, SERVER_PORT))
print(f"[*] Connected to {SERVER_HOST}:{SERVER_PORT}")

while True:
    # 서버로 데이터 전송
    input_data = input("input : ")
    client_socket.sendall(input_data.encode())

    # 서버로부터 데이터 수신
    # data = client_socket.recv(1024)
    # print(f"[*] Received data: {data.decode()}")

# 연결 종료 
client_socket.close()
