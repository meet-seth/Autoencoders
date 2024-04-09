import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--one"
)

parser.add_argument(
    "--two"
)

args = parser.parse_args()

print(args)

print(vars(args))
d = {}
d['one'] = vars(args).pop("one")

print(vars(args))
print(d)