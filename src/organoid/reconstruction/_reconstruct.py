def reconstruct_foo():
    print("reconstruct")
    return -1


def pipeline_reconstruction(*args):  # image1, image2, sigma, ...
    print(*args)


def script_run():

    # Parse the arguments
    pipeline_reconstruction(1, 2, 3, 4, 5)
