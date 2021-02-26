""" TODO: launch the 'sequoia gRPC server' at a given address / port. """
import argparse
from .server import server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ip", type=str, help="gRPC host ip", default="")
    parser.add_argument("-p", "--port", type=int, help="gRPC port", default=13337)
    args = parser.parse_args()

    server(
        grpc_host=args.ip,
        grpc_port=args.port,
    )
