import re
import subprocess

from tqdm import tqdm
from defects4j_get_method import df4j_get_method

df4g_gm = df4j_get_method()
repo_list = ["Cli", "Closure", "Codec", "Collections", "Compress", "Csv", "Gson", "JacksonCore",
             "JacksonDatabind", "JacksonXml", "Jsoup", "JxPath", "Lang", "Math", "Mockito", "Time"]

for repo_name in tqdm(repo_list, desc="Processing"):
    result = subprocess.run(["defects4j", "query", "-p", repo_name,
                            "-q", "revision.id.fixed"], stdout=subprocess.PIPE)

    # 35,085a1538fa20d8e48faad49eaffd697f024bf1af
    # 36,afc13c445a4c80432e52d735685b272fadfeeddf
    # 37,1bf9e6c551b6a2e7d37291673a1ff77c338ce131
    # 38,ac2a1c85616f0140418de9190389fe7b80296c39
    # 39,0b453953fa5f55cf2e8fd034d4d55972deb7647a
    # 40,b0024d482050a08efc36c3cabee37c0af0e57a10

    # convert to a list of tuples:
    result = result.stdout.decode("utf-8")
    result = result.split("\n")
    result = [x for x in result if x]
    result = [re.split(r",", x) for x in result]
    id2cmtsha = [(int(x[0]), x[1]) for x in result]
    print(id2cmtsha)
    for idx, cmtsha in tqdm(id2cmtsha):
        print(f"Processing {repo_name} {idx}")
        subprocess.run(["defects4j", "checkout", "-p", repo_name, "-v",
                       f"{idx}f", "-w", f"./tmp/{repo_name}-{idx}"], capture_output=False, text=True)
        df4g_gm.run(f"./tmp/{repo_name}-{idx}", cmtsha)
        # remove the repo
        subprocess.run(
            ["rm", "-rf", f"./tmp/{repo_name}-{idx}"], capture_output=False, text=True)
