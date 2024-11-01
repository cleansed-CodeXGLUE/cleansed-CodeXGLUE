package org.example;

import com.github.gumtreediff.actions.ChawatheScriptGenerator;
import com.github.gumtreediff.actions.EditScript;
import com.github.gumtreediff.actions.EditScriptGenerator;
import com.github.gumtreediff.actions.SimplifiedChawatheScriptGenerator;
import com.github.gumtreediff.actions.model.Action;
import com.github.gumtreediff.client.Run;
import com.github.gumtreediff.gen.TreeGenerators;
import com.github.gumtreediff.matchers.MappingStore;
import com.github.gumtreediff.matchers.Matcher;
import com.github.gumtreediff.matchers.Matchers;
import com.github.gumtreediff.tree.Tree;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class get_ept_actions {
    public static void main(String[] args) throws IOException {
        Run.initGenerators(); // registers the available parsers
        String src_root_path = "/home/shweng/code_data_clean/code-refinement/data/small-addMain/train-buggy-javafile/";
        String dst_root_path = "/home/shweng/code_data_clean/code-refinement/data/small-addMain/train-fixed-javafile/";

        String org_src_root_path = "/home/shweng/code_data_clean/code-refinement/data/small-addMain/org-train-buggy-javafile/";
        String org_dst_root_path = "/home/shweng/code_data_clean/code-refinement/data/small-addMain/org-train-fixed-javafile/";

        String output_ept_buggy_path = "/home/shweng/code_data_clean/code-refinement/data/small-addMain/train-empty-buggy-action.json";
        String output_ept_fixed_path = "/home/shweng/code_data_clean/code-refinement/data/small-addMain/train-empty-fixed-action.json";

        String empty_path = "/home/shweng/code_data_clean/code-refinement/data/small-addMain/empty.java";
        File src_directory = new File(src_root_path);
        File dst_directory = new File(dst_root_path);
        File org_src_directory = new File(org_src_root_path);
        File org_dst_directory = new File(org_dst_root_path);

        File[] src_files = src_directory.listFiles();
        File[] dst_files = dst_directory.listFiles();
        File[] org_src_files = org_src_directory.listFiles();
        File[] org_dst_files = org_dst_directory.listFiles();

        int length = src_files.length;
        assert length == dst_files.length;
        System.out.println("length: " + length);
        JSONObject jsonActions = new JSONObject();
        for (int k = 0; k < 2; k++) {
            for (int i = 0; i < length; i++) {

                String srcFile = new String();
                String dstFile = new String();
                String org_dstFile = new String();
                
                if (k == 0) {
                    // empty - buggy
                    srcFile = empty_path;
                    dstFile = src_files[i].getAbsolutePath();
                    org_dstFile = org_src_files[i].getAbsolutePath();
                }
                else if (k == 1) {
                    // empty - fixed
                    srcFile = empty_path;
                    dstFile = dst_files[i].getAbsolutePath();
                    org_dstFile = org_dst_files[i].getAbsolutePath();
                }
                
                System.out.println("Processing file " + i + " of " + length + ":" + srcFile);
                
                // ==================cur=================
                Tree src = TreeGenerators.getInstance().getTree(srcFile).getRoot(); 
                Tree dst = TreeGenerators.getInstance().getTree(dstFile).getRoot(); 
                Matcher defaultMatcher = Matchers.getInstance().getMatcher(); 
                MappingStore mappings = defaultMatcher.match(src, dst); 
                EditScriptGenerator editScriptGenerator = new ChawatheScriptGenerator(); 
                EditScript actions = editScriptGenerator.computeActions(mappings); 
                // array of actions
                JSONArray opts = new JSONArray();
                JSONArray fulls = new JSONArray();

                for (Action action : actions) {
                    opts.put(action.getName());
                    fulls.put(action.toString());
                }
                // ======================================
                
                // ==================org=================
                Tree src2 = TreeGenerators.getInstance().getTree(srcFile).getRoot();
                Tree dst2 = TreeGenerators.getInstance().getTree(org_dstFile).getRoot();
                Matcher defaultMatcher2 = Matchers.getInstance().getMatcher();
                MappingStore mappings2 = defaultMatcher2.match(src2, dst2);
                EditScriptGenerator editScriptGenerator2 = new ChawatheScriptGenerator();
                EditScript action_org = editScriptGenerator2.computeActions(mappings2);
                // array of actions
                JSONArray opts_org = new JSONArray();
                JSONArray fulls_org = new JSONArray();

                for (Action action : action_org) {
                    opts_org.put(action.getName());
                    fulls_org.put(action.toString());
                }
                
                JSONObject actionJson = new JSONObject();
                // idx is the java file name without the extension
                int idx = Integer.parseInt(src_files[i].getName().split("\\.")[0]);
                actionJson.put("idx", String.valueOf(idx));
                actionJson.put("opts", opts);
                actionJson.put("fulls", fulls);
                actionJson.put("opts_org", opts_org);
                actionJson.put("fulls_org", fulls_org);
                jsonActions.put(String.valueOf(idx), actionJson);
            }
            if (k == 0) {
                try (FileWriter file = new FileWriter(output_ept_buggy_path)) {
                    file.write(jsonActions.toString(4)); // pretty print with an indent of 4
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            else if (k == 1) {
                try (FileWriter file = new FileWriter(output_ept_fixed_path)) {
                    file.write(jsonActions.toString(4)); // pretty print with an indent of 4
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        

    }
}