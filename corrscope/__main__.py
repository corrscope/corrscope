from corrscope import cli
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    cli.main()
