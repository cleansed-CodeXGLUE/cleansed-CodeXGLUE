import subprocess
import javalang
import os


class df4j_get_method_xvn:

    def __init__(self):
        self.global_cnt = 0

    def get_commit_diff(self, repo_path, commit_revision):
        cmd = ['svn', 'diff', '-c', commit_revision, repo_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout

    def extract_methods(self, java_code):
        try:
            tree = javalang.parse.parse(java_code)
        except javalang.parser.JavaSyntaxError as e:
            print(f"JavaSyntaxError: {e.description} at {e.at}")
            print("Problematic Java code:")
            print(java_code)
            return {}

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

    def save_methods(self, methods, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for method_name, method_content in methods.items():
            file_name = os.path.join(folder_path, f"{self.global_cnt}.java")
            with open(file_name, 'w') as f:
                f.write(method_content)
            self.global_cnt += 1

    def get_file_content(self, repo_path, file_path, revision):
        cmd = ['svn', 'cat', '-r', revision,
               os.path.join(repo_path, file_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout

    def filter_java_code(self, diff_content):
        lines = diff_content.split('\n')
        java_lines = []
        for line in lines:
            if line.startswith('@@') or line.startswith('Index:') or line.startswith('---') or line.startswith('+++'):
                continue
            if line.startswith('-') or line.startswith('+'):
                java_lines.append(line[1:])
            else:
                java_lines.append(line)
        return '\n'.join(java_lines)

    def run(self, repo_path, commit_revision):
        diff = self.get_commit_diff(repo_path, commit_revision)

        lines = diff.split('\n')
        current_file = None
        old_content = []
        new_content = []
        old_revision = str(int(commit_revision) - 1)

        for line in lines:
            if line.startswith('Index: '):
                if current_file and current_file.endswith('.java'):
                    old_content_str = self.filter_java_code(
                        '\n'.join(old_content))
                    new_content_str = self.filter_java_code(
                        '\n'.join(new_content))

                    methods_before = self.extract_methods(old_content_str)
                    methods_after = self.extract_methods(new_content_str)

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

                    self.save_methods(changed_methods_before, before_folder)
                    self.save_methods(changed_methods_after, after_folder)

                current_file = line.split(' ')[1]
                old_content = []
                new_content = []
            elif line.startswith('--- '):
                pass
            elif line.startswith('+++ '):
                pass
            elif line.startswith('-'):
                old_content.append(line)
            elif line.startswith('+'):
                new_content.append(line)
            else:
                old_content.append(line)
                new_content.append(line)

        if current_file and current_file.endswith('.java'):
            old_content_str = self.filter_java_code('\n'.join(old_content))
            new_content_str = self.filter_java_code('\n'.join(new_content))

            methods_before = self.extract_methods(old_content_str)
            methods_after = self.extract_methods(new_content_str)

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

            self.save_methods(changed_methods_before, before_folder)
            self.save_methods(changed_methods_after, after_folder)

# Example usage:
# df4j = df4j_get_method()
# df4j.run('/path/to/svn/repo', '123')
