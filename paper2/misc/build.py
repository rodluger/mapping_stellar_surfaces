import glob
import subprocess
import os
import json
from tqdm import tqdm
import pytest
import sys
import shutil


DATA_URL = "https://users.flatironinstitute.org/~rluger/starry_process/data_paper2.tar.gz"


def generate_github_links():
    try:
        HASH = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode(
            "utf-8"
        )[:-1]
    except:
        HASH = "master"

    try:
        SLUG = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"]
        ).decode("utf-8")[19:-1]
    except:
        SLUG = "user/repo"

    with open("gitlinks.tex", "w") as f:
        print(
            r"\newcommand{\codelink}[1]{\href{https://github.com/%s/blob/%s/paper2/figures/#1.py}{\codeicon}\,\,}"
            % (SLUG, HASH),
            file=f,
        )
        print(
            r"\newcommand{\animlink}[1]{\href{https://github.com/%s/blob/%s/paper2/figures/#1.gif}{\animicon}\,\,}"
            % (SLUG, HASH),
            file=f,
        )
        print(
            r"\newcommand{\prooflink}[1]{\href{https://github.com/%s/blob/%s/paper2/tests/#1.py}{\raisebox{-0.1em}{\input{tests/#1.tex}}}}"
            % (SLUG, HASH),
            file=f,
        )
        print(
            r"\newcommand{\cilink}[1]{\href{https://dev.azure.com/%s/_build}{#1}}"
            % (SLUG),
            file=f,
        )


def download_data(clobber=False):
    if clobber or not os.path.exists("figures/data"):

        # Download the tarball
        subprocess.check_output(["wget", DATA_URL, "-O", "data.tar.gz"])

        # Extract it
        subprocess.check_output(
            ["tar", "-xzvf", "data.tar.gz", "-C", "figures"]
        )

        # Remove the tarball
        os.remove("data.tar.gz")


def build_figures():

    # Get all figure scripts
    scripts = glob.glob("figures/*.py")

    # Get metadata file
    if not os.path.exists("figures/metadata.json"):
        meta = {}
    else:
        with open("figures/metadata.json", "r") as f:
            meta = json.load(f)

    # Run each figure if the script changed, if
    # the output is missing, or if there was an
    # error last time we ran it
    iterator = tqdm(scripts)
    for script in iterator:
        name = script.replace("figures/", "")
        iterator.set_description("Running {}".format(name))
        st_mtime = os.stat(script).st_mtime
        stale = meta.get(name, {}).get("st_mtime", 0) != st_mtime
        missing = not all(
            [
                os.path.exists(output)
                for output in meta.get(name, {}).get("outputs", [])
            ]
        )
        error = bool(len(meta.get(name, {}).get("stderr", "")))
        if stale or missing or error:
            old_figures = set(
                glob.glob("figures/*.pdf") + glob.glob("figures/*.png")
            )
            try:
                stdout = subprocess.check_output(
                    ["python", script],
                    env=dict(os.environ, CI="1", NOTQDM="1"),
                    stderr=subprocess.STDOUT,
                ).decode("utf-8")
                stderr = ""
            except subprocess.CalledProcessError as e:
                stdout = ""
                stderr = e.output.decode("utf-8")
                # Fail silently
                print("ERROR running {}:".format(name))
                print(stderr)

            new_figures = set(
                glob.glob("figures/*.pdf") + glob.glob("figures/*.png")
            )
            outputs = list(new_figures - old_figures)
            meta[name] = dict(
                st_mtime=st_mtime,
                outputs=outputs,
                stdout=stdout,
                stderr=stderr,
            )

    # Update the metadata
    with open("figures/metadata.json", "w") as f:
        json.dump(meta, f)


def run_tests():

    # Get all test files
    test_files = glob.glob("tests/test_*.py")

    # Get metadata file
    if not os.path.exists("tests/metadata.json"):
        meta = {}
    else:
        with open("tests/metadata.json", "r") as f:
            meta = json.load(f)

    # Run each test if the script changed, if
    # the output is missing, or if there was an
    # error last time we ran it
    total_passed = 0
    total_failed = 0
    iterator = tqdm(test_files)
    os.environ["NOTQDM"] = "1"
    for test_file in iterator:
        name = test_file.replace("tests/", "")
        iterator.set_description("Running {}".format(name))
        st_mtime = os.stat(test_file).st_mtime
        log_file = test_file.replace(".py", ".tex")
        stale = meta.get(name, {}).get("st_mtime", 0) != st_mtime
        missing = not os.path.exists(
            "tests/{}".format(name.replace(".py", ".tex"))
        )
        passed = meta.get(name, {}).get("passed", 0)
        if stale or missing or not passed:
            result = pytest.main([test_file, "-qq"]).value
            with open(log_file, "w") as f:
                if result == 0:
                    print(r"\testpassicon", file=f)
                    passed = 1
                    total_passed += 1
                else:
                    print(r"\testfailicon", file=f)
                    passed = 0
                    total_failed += 1
            meta[name] = dict(st_mtime=st_mtime, passed=passed)
        else:
            total_passed += 1

    # Update the metadata
    with open("tests/metadata.json", "w") as f:
        json.dump(meta, f)

    # Tally the results
    with open("tests/tally.tex", "w") as f:
        print(
            r"{} passed \testpassicon, {} failed \testfailicon".format(
                total_passed, total_failed
            ),
            file=f,
        )


def build_pdf():
    subprocess.check_output(
        ["tectonic", "ms.tex", "--keep-logs", "--keep-intermediates"]
    )


def build():
    generate_github_links()
    run_tests()
    download_data()
    build_figures()
    build_pdf()


def clean(remove_data=False):
    # Remove figure output
    for file in glob.glob("figures/*.pdf") + glob.glob("figures/*.png"):
        os.remove(file)
    # Remove test output
    for file in glob.glob("tests/*.tex"):
        os.remove(file)
    # Remove all metadata
    for file in glob.glob("*/metadata.json"):
        os.remove(file)
    # Remove TeX files and temporary files
    for pattern in [
        "*.pdf",
        "*.png",
        "*.bbl",
        "*.blg",
        "*.log",
        "*.aux",
        "gitlinks.tex",
    ]:
        for file in glob.glob(pattern):
            os.remove(file)
    # Remove cached data
    if remove_data and os.path.exists("figures/data"):
        shutil.rmtree("figures/data")


if __name__ == "__main__":
    build()
