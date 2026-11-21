import logging

from utils.program import Program


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    Program().run()


if __name__ == "__main__":
    main()
