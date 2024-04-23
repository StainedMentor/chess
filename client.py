import socket
import threading



class Client:
    """
    This class automatically sets up a server or client on the specifed address.
    Receive_messages function can be modified to pass the message to the desired location
    Send_text_message function send the given string to the server/client it is connected to
    """
    def __init__(self, host='127.0.0.1', port=12347, buffer=None, log=None, cmd=None):
        self.host = host
        self.port = port
        self.buffer = buffer
        self.log = log
        self.isClient = False
        self.cmd = cmd
        self.initialise_connection()

    def initialise_connection(self):
        try:
            self.start_as_server()
        except:
            self.isClient = True
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))
            threading.Thread(target=self.receive_messages).start()


    def start_as_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        threading.Thread(target=self.accept_connections).start()

    def accept_connections(self):
        while True:
            try:
                client_socket, addr = self.server_socket.accept()
                self.client_socket = client_socket

                threading.Thread(target=self.receive_messages).start()
            except Exception as e:
                pass
                print(f"An error occurred: {e}")

    def receive_messages(self):
        while True:
            try:
                message = self.client_socket.recv(1024).decode()
                if not message:
                    break

                if message.startswith("MOVE|"):
                    message = message[5:]
                    # self.buffer.put(message)
                    print(message)
                    self.cmd(message)

                if message.startswith("MSG|"):
                    message = message[4:]
                    self.log.put(message)


            except Exception as e:
                break



    def send_text_message(self, message):
        try:
            self.client_socket.send(message.encode())
        except Exception as e:
            pass
            print(f"An error occurred while sending the text message: {e}")

    def close(self):
        self.client_socket.close()


