import git
import javalang
import os


class df4j_get_method:

    def __init__(self):
        self.global_cnt = 0

    def get_commit_diff(self, repo_path, commit_sha):
        repo = git.Repo(repo_path)
        commit = repo.commit(commit_sha)
        parent_commit = commit.parents[0]
        diff = None
        try:
            diff = parent_commit.diff(commit, create_patch=True)
        except:
            print(f"Error in getting diff for {commit_sha}")
            diff = []
        return diff

    def extract_methods(self, java_code):
        tree = javalang.parse.parse(java_code)
        lines = java_code.splitlines()
        methods = {}

        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            start_line = node.position.line - 1
            method_code_lines = [lines[start_line]]

            # Collect lines until the end of the method
            brace_count = method_code_lines[0].count(
                '{') - method_code_lines[0].count('}')
            end_line = start_line + 1

            while brace_count > 0 and end_line < len(lines):
                line = lines[end_line]
                method_code_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                end_line += 1

            method_code = '\n'.join(method_code_lines)
            methods[node.name] = method_code

        return methods

    def save_methods(self, changed_methods_before, before_folder,
                     changed_methods_after, after_folder):
        if not os.path.exists(before_folder):
            os.makedirs(before_folder)
        if not os.path.exists(after_folder):
            os.makedirs(after_folder)

        for method_before, method_after in zip(changed_methods_before.items(), changed_methods_after.items()):
            method_name_before, method_content_before = method_before
            method_name_after, method_content_after = method_after

            file_name = os.path.join(before_folder, f"{self.global_cnt}.java")
            with open(file_name, 'w') as f:
                f.write(method_content_before)

            file_name = os.path.join(after_folder, f"{self.global_cnt}.java")
            with open(file_name, 'w') as f:
                f.write(method_content_after)

            self.global_cnt += 1

    def get_blob_content(self, blob):
        # if utf-8 does not work, try ISO-8859-1
        try:
            res = blob.data_stream.read().decode('utf-8')
        except:
            res = blob.data_stream.read().decode('ISO-8859-1')

        return res

    def run(self, repo_path, commit_sha):
        diff = self.get_commit_diff(repo_path, commit_sha)

        for diff_entry in diff:
            # bug: AttributeError: 'NoneType' object has no attribute 'endswith'
            if diff_entry is None:
                continue
            if diff_entry.a_path is None:
                continue
            if diff_entry.a_path.endswith('.java'):
                # if diff_entry.a_path.endswith('.java'):
                old_blob = diff_entry.a_blob
                new_blob = diff_entry.b_blob

                old_content = self.get_blob_content(
                    old_blob) if old_blob else ''
                new_content = self.get_blob_content(
                    new_blob) if new_blob else ''

                methods_before = self.extract_methods(old_content)
                methods_after = self.extract_methods(new_content)

                # Find common methods
                common_methods = set(methods_before.keys()).intersection(
                    set(methods_after.keys()))

                # Filter methods to only include common ones and check for changes
                changed_methods_before = {
                    name: methods_before[name] for name in common_methods if methods_before[name] != methods_after[name]}
                changed_methods_after = {
                    name: methods_after[name] for name in common_methods if methods_before[name] != methods_after[name]}

                before_folder = 'before'
                after_folder = 'after'

                self.save_methods(changed_methods_before, before_folder,
                                  changed_methods_after, after_folder)
