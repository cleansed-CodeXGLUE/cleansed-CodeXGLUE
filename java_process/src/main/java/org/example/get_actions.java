package org.example;

import com.github.gumtreediff.actions.ChawatheScriptGenerator;
import com.github.gumtreediff.actions.EditScript;
import com.github.gumtreediff.actions.EditScriptGenerator;
import com.github.gumtreediff.actions.model.Action;
import com.github.gumtreediff.client.*;
import com.github.gumtreediff.gen.TreeGenerators;
import com.github.gumtreediff.matchers.MappingStore;
import com.github.gumtreediff.matchers.Matcher;
import com.github.gumtreediff.matchers.Matchers;
import com.github.gumtreediff.tree.Tree;

import org.json.JSONArray;
import org.json.JSONObject;
import java.io.FileWriter;
import java.io.IOException;
import java.io.File;

public class get_actions {
    public static void main(String[] args) throws IOException {
        Run.initGenerators(); // registers the available parsers
        String src_root_path = "/home/shweng/code_data_clean/code-refinement/data/small-addMain/train-buggy-javafile/";
        String dst_root_path = "/home/shweng/code_data_clean/code-refinement/data/small-addMain/train-fixed-javafile/";
        String org_src_root_path = "/home/shweng/code_data_clean/code-refinement/data/small-addMain/org-train-buggy-javafile/";
        String org_dst_root_path = "/home/shweng/code_data_clean/code-refinement/data/small-addMain/org-train-fixed-javafile/";

        String output_path = "/home/shweng/code_data_clean/code-refinement/data/small-addMain/train-action.json";
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
        for (int i = 0; i < length; i++) {
            // progress bar

            String srcFile = src_files[i].getAbsolutePath();
            String dstFile = dst_files[i].getAbsolutePath();
            String org_srcFile = org_src_files[i].getAbsolutePath();
            String org_dstFile = org_dst_files[i].getAbsolutePath();
            System.out.println("Processing file " + i + " of " + length+ ":" + srcFile);

            // ==================cur=================
            Tree src = TreeGenerators.getInstance().getTree(srcFile).getRoot(); // retrieves and applies the default parser for the file
            Tree dst = TreeGenerators.getInstance().getTree(dstFile).getRoot(); // retrieves and applies the default parser for the file

            Matcher defaultMatcher = Matchers.getInstance().getMatcher(); // retrieves the default matcher
            MappingStore mappings = defaultMatcher.match(src, dst); // computes the mappings between the trees
            EditScriptGenerator editScriptGenerator = new ChawatheScriptGenerator(); // instantiates the simplified Chawathe script generator
            EditScript actions = editScriptGenerator.computeActions(mappings); // computes the edit script

            // array of actions
            JSONArray opts = new JSONArray();
            JSONArray fulls = new JSONArray();
            for (Action action : actions) {
                opts.put(action.getName());
                fulls.put(action.toString());
            }

            // rev
            Tree src2 = TreeGenerators.getInstance().getTree(srcFile).getRoot(); // retrieves and applies the default parser for the file
            Tree dst2 = TreeGenerators.getInstance().getTree(dstFile).getRoot();
            Matcher defaultMatcher2 = Matchers.getInstance().getMatcher(); // retrieves the default matcher
            MappingStore mappings2 = defaultMatcher2.match(dst2, src2); // computes the mappings between the trees
            EditScriptGenerator editScriptGenerator2 = new ChawatheScriptGenerator(); // instantiates the simplified Chawathe script generator
            EditScript actions_rev = editScriptGenerator2.computeActions(mappings2); // computes the edit script

            // array of actions rev
            JSONArray opts_rev = new JSONArray();
            JSONArray fulls_rev = new JSONArray();
            for (Action action : actions_rev) {
                opts_rev.put(action.getName());
                fulls_rev.put(action.toString());
            }
            // ======================================

            // ==================org=================
            Tree org_src = TreeGenerators.getInstance().getTree(org_srcFile).getRoot(); // retrieves and applies the default parser for the file
            Tree org_dst = TreeGenerators.getInstance().getTree(org_dstFile).getRoot(); // retrieves and applies the default parser for the file

            Matcher defaultMatcher_org = Matchers.getInstance().getMatcher(); // retrieves the default matcher
            MappingStore mappings_org = defaultMatcher_org.match(org_src, org_dst); // computes the mappings between the trees
            EditScriptGenerator editScriptGenerator_org = new ChawatheScriptGenerator(); // instantiates the simplified Chawathe script generator
            EditScript actions_org = editScriptGenerator_org.computeActions(mappings_org); // computes the edit script

            // array of actions
            JSONArray opts_org = new JSONArray();
            JSONArray fulls_org = new JSONArray();
            for (Action action : actions_org) {
                opts_org.put(action.getName());
                fulls_org.put(action.toString());
            }

            // rev
            Tree org_src2 = TreeGenerators.getInstance().getTree(org_srcFile).getRoot(); // retrieves and applies the default parser for the file
            Tree org_dst2 = TreeGenerators.getInstance().getTree(org_dstFile).getRoot();
            Matcher defaultMatcher_org2 = Matchers.getInstance().getMatcher(); // retrieves the default matcher
            MappingStore mappings_org2 = defaultMatcher_org2.match(org_dst2, org_src2); // computes the mappings between the trees
            EditScriptGenerator editScriptGenerator_org2 = new ChawatheScriptGenerator(); // instantiates the simplified Chawathe script generator
            EditScript actions_org_rev = editScriptGenerator_org2.computeActions(mappings_org2); // computes the edit script
            
            // array of actions rev
            JSONArray opts_org_rev = new JSONArray();
            JSONArray fulls_org_rev = new JSONArray();
            for (Action action : actions_org_rev) {
                opts_org_rev.put(action.getName());
                fulls_org_rev.put(action.toString());
            }

            // ======================================


            JSONObject actionJson = new JSONObject();
            // idx is the java file name without the extension
            int idx = Integer.parseInt(src_files[i].getName().split("\\.")[0]);
            actionJson.put("idx", String.valueOf(idx));
            actionJson.put("opts", opts);
            actionJson.put("fulls", fulls);
            actionJson.put("opts_rev", opts_rev);
            actionJson.put("fulls_rev", fulls_rev);
            actionJson.put("opts_org", opts_org);
            actionJson.put("fulls_org", fulls_org);
            actionJson.put("opts_org_rev", opts_org_rev);
            actionJson.put("fulls_org_rev", fulls_org_rev);
            jsonActions.put(String.valueOf(idx),actionJson);
        }
        try (FileWriter file = new FileWriter(output_path)) {
           file.write(jsonActions.toString(4)); // pretty print with an indent of 4
       } catch (IOException e) {
           e.printStackTrace();
       }
        
    }
}