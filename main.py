"""
main.py

Welcome to the course Data Science Advanced! This script will run some scripts using the machine learning framework
 to check if they are installed properly. It is likely that the Tensorflow core will raise some warnings to help you
 get the most out of your hardware and configuration.

@author Lina Blijleven <selina.blijleven@code-cafe.nl>
"""
import installation_tests.keras_test
import installation_tests.tensorflow_test
import installation_tests.torch_test


def run():
    print(f"Welcome to the course!")
    fails: bool = False

    try:
        installation_tests.keras_test.run()
    except:
        fails = True
        print(f"Keras test failed :(")

    try:
        installation_tests.torch_test.run()
    except:
        fails = True
        print(f"PyTorch test failed :(")

    try:
        installation_tests.tensorflow_test.run()
    except:
        fails = True
        print("PyTorch test failed :(")

    if not fails:
        print("Everything seems to be installed! "
              "The libraries might show some different warnings to guide you in their usage.")


if __name__ == '__main__':
    run()
