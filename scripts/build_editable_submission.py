"""Create editable PowerPoint and Word deliverables from actual saved metrics.

Requires python-pptx and python-docx. No results are invented for pending runs.
"""
import argparse
import csv
import json
import pathlib
import zipfile

from docx import Document
from docx.shared import Inches as DocInches, Pt as DocPt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.chart.data import CategoryChartData
from pptx.util import Inches, Pt


def load(path): return json.loads(path.read_text())


def gpu_snapshot(root):
    folder=root/"xlmr_mixed_risk"
    if not (folder/"verification.json").exists(): return None
    status=load(folder/"verification.json")
    if status.get("status")!="verified completed run; metrics recomputed":
        raise ValueError("GPU metrics have not passed the result importer")
    return dict(metrics=load(folder/"verified_metrics.json"),run=load(folder/"run.json"),status=status)


def collect(root):
    rows=[]
    for target in ["risk","action"]:
        majority=root/"majority_baselines.json"
        if majority.exists():
            fixed=load(majority)[target]
            for language in ["test_en","test_kk"]:
                rows.append(dict(target=target,model="Training-majority baseline",language=language,metrics=fixed[language]))
        base=load(root/f"{target}_tfidf/baselines.json")
        for name,key in [("Word TF-IDF, English","word_en"),("Word TF-IDF, mixed","word_mixed")]:
            for lang,m in base[key].items(): rows.append(dict(target=target,model=name,language=lang,metrics=m))
        tuned=load(root/"tuned_sparse/test_metrics.json")[target]
        for lang,m in tuned.items(): rows.append(dict(target=target,model="Sparse selected on dev",language=lang,metrics=m))
        for suffix,label in [("mixed_seed42","Word128"),("hybrid512_seed42","Hybrid512"),("semantic_seed42","LaBSE + 54 text features")]:
            path=root/f"diffusion_{target}_{suffix}/test_metrics.json"
            if not path.exists(): continue
            data=load(path)
            models=["linear","mlp","diffusion"] if suffix=="semantic_seed42" else ["mlp","diffusion"]
            for lang,result in data.items():
                for model in models: rows.append(dict(target=target,model=f"{label}: {model}",language=lang,metrics=result[model]))
    gpu=gpu_snapshot(root)
    if gpu:
        for lang,metrics in gpu["metrics"].items():
            rows.append(dict(target="risk",model="XLM-R mixed (GPU verified)",language=lang,metrics=metrics))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=pathlib.Path,default=pathlib.Path("data/experiments/deadline_2026-11-23"))
    args=parser.parse_args(); root=args.runs
    out=root/"submission"; out.mkdir(exist_ok=True)
    rows=collect(root)
    gpu=gpu_snapshot(root)
    progress=load(root/"xlmr_progress.json") if (root/"xlmr_progress.json").exists() else None
    gpu_status="verified GPU results included" if gpu else "GPU evaluation pending"
    stability_path=root/"seed_stability/summary.json"
    stability=load(stability_path) if stability_path.exists() else None
    with (out/"all_metrics.csv").open("w",newline="") as f:
        fields=["target","model","language","macro_f1","supported_macro_f1","accuracy","binary_macro_f1"]
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in rows:
            m=r["metrics"]
            writer.writerow(dict(target=r["target"],model=r["model"],language=r["language"],macro_f1=m["macro_f1"],
                supported_macro_f1=m["macro_f1_supported"],accuracy=m["accuracy"],binary_macro_f1=m.get("binary_macro_f1","")))
    def table_rows(target):
        names=list(dict.fromkeys(r["model"] for r in rows if r["target"]==target))
        lookup={(r["model"],r["language"]):r["metrics"] for r in rows if r["target"]==target}
        key="macro_f1" if target=="risk" else "macro_f1_supported"
        return [[name,f"{lookup[name,'test_en'][key]:.4f}",f"{lookup[name,'test_kk'][key]:.4f}"] for name in names]
    references=[
        "[1] Source corpus: shaw/scambench-training. https://huggingface.co/datasets/shaw/scambench-training (local processed snapshot: 37,421 original rows).",
        "[2] Han, X., Zheng, H., and Zhou, M. (2022). CARD: Classification and Regression Diffusion Models. NeurIPS. https://arxiv.org/abs/2206.07275",
        "[3] Feng, F., Yang, Y., Cer, D., Arivazhagan, N., and Wang, W. (2022). Language-agnostic BERT Sentence Embedding. ACL. https://aclanthology.org/2022.acl-long.62/",
        "[4] FacebookAI/xlm-roberta-base model card. https://huggingface.co/FacebookAI/xlm-roberta-base",
    ]
    # Word report: native text, tables and headings remain editable in Word.
    doc=Document()
    normal=doc.styles["Normal"]; normal.font.name="Calibri"; normal.font.size=DocPt(11)
    normal.paragraph_format.space_after=DocPt(7)
    section=doc.sections[0]; section.top_margin=DocInches(.75); section.bottom_margin=DocInches(.75)
    doc.add_heading("ScamBench-KK",0)
    doc.add_paragraph("Kazakh extension, risk classification, action prediction and conditional diffusion",style="Subtitle")
    doc.add_paragraph("Results snapshot: 23 November 2026\nAuthor / course: [complete before submission]\nStatus: completed CPU experiments; "+gpu_status)
    doc.add_heading("Abstract",1)
    abstract="This project extends a processed ScamBench snapshot with a three-class risk taxonomy and 300 Kazakh reference conversations. We investigate cross-lingual classification, identify conversation-family leakage in the original splits, and evaluate English-only and English-plus-silver training. A further task predicts the original agent action from conversation text. We implement a CARD-inspired conditional label-diffusion model and compare it with matched linear and neural classifiers. Word TF-IDF trained with silver Kazakh reaches 0.6675 three-class and 0.8983 merged-binary macro-F1 on the current Kazakh benchmark. Development-selected sparse tuning gives 0.4328 action macro-F1 over the five supported test actions. Feature follow-ups improve some neural point estimates but also produce regressions. Results remain exploratory: limited seed coverage, a reused historical benchmark and unreviewed Kazakh constrain general conclusions. "
    abstract+=(f"Verified XLM-R Kazakh test macro-F1 is {gpu['metrics']['test_kk']['macro_f1']:.4f}." if gpu else "End-to-end XLM-R test results remain pending and are not claimed here.")
    doc.add_paragraph(abstract)
    doc.add_heading("1. Dataset and targets",1)
    doc.add_paragraph("The local processed source snapshot contains 37,421 multi-turn conversations [1]. The merged extension contains 37,721 rows after adding 300 Kazakh conversations. The selected Kazakh slice is balanced at 100 conversations per risk class. Each translation retains its English source label and conversation identity. Automatic validation checks structure and entities; native-speaker review of the whole set has not been completed.")
    doc.add_paragraph("The derived risk label combines the original attack-defense flag with the prescribed decision. SCAM corresponds to an attack requiring a defensive action; SUSPICIOUS covers verification/audit or other borderline risk; LEGITIMATE covers normal interaction without an attack signal. These are rule-derived annotations, not independently adjudicated human diagnoses. The merged-binary task combines SCAM and SUSPICIOUS predictions as ATTACK; its score must not be presented as a three-class result.")
    doc.add_paragraph("Action prediction uses the original decision_class as its output target. After spelling normalization there are thirteen action types. Inputs contain conversation text, not the action, recorded response, explanation, risk indicators or other answer metadata. This is offline prediction of recorded decisions, not reinforcement learning, action execution or a measurement of policy safety.")
    doc.add_heading("2. Leakage control and evaluation",1)
    doc.add_paragraph("The source corpus contains multiple restylings of conversations. The project audit found that 83% of original validation rows had a conversation-family counterpart in training. Family-based partitioning keeps restylings together; translated copies explicitly follow their sources. In the current paired experiment, all families represented by the Kazakh evaluation slice are removed from the English training pool.")
    t=doc.add_table(rows=1,cols=3); t.style="Light Shading Accent 1"
    for c,v in zip(t.rows[0].cells,["Partition","English rows","Kazakh rows"]): c.text=v
    for values in [["Training","11,227","3,600 silver"],["Development","1,208","399 silver"],["Historical evaluation","300","300 reference"]]:
        for c,v in zip(t.add_row().cells,values): c.text=v
    doc.add_paragraph("The evaluation pairs belong to 273 distinct conversation families. Training, development and evaluation families are disjoint. The historical benchmark has already informed project decisions; these follow-ups are not a fresh confirmatory evaluation. All learned text transformations and standardization fit only training data. Frozen semantic features are computed without labels. Checkpoints and sparse hyperparameters use development data.")
    duplicate_audit=root/"full_text_duplicate_audit.json"
    if duplicate_audit.exists():
        audit=load(duplicate_audit)["exact"]
        doc.add_paragraph(f"A subsequent full-text check found {audit['development']['overlapping_rows']} of {audit['development']['rows']} development rows identical to training text despite distinct family IDs. This is a remaining development-selection caveat, not proof of a particular score inflation. The active protocol was not changed retrospectively. The paired test has {audit['test']['overlapping_rows']} exact training duplicates among {audit['test']['rows']} rows. Whitespace normalization gives the same counts; near-duplicates remain a limitation.")
    doc.add_paragraph("Risk results use three-class macro-F1 and report merged-binary macro-F1 separately. The trained action schema has thirteen classes, but the evaluation slice contains only refuse (99), engage_legitimate (88), request_verification (87), audit (25), and ignore (1). Main action tables therefore report macro-F1 over those five supported classes; full-schema scores and support remain in the raw metrics. This is limited action coverage, not evaluation of all thirteen decisions.")
    doc.add_heading("3. Models and development selection",1)
    doc.add_paragraph("Word and character TF-IDF baselines use linear SVMs. English-only training is compared with English plus 3,600 existing silver Kazakh rows. A fixed sparse search tests three feature choices, C in {0.25, 1, 4}, and optional inverse-frequency class weights: eighteen candidates per target. Selection maximizes silver-Kazakh development macro-F1, using pooled development scores to break ties. It selects hybrid features with C=4 for risk, and balanced word features with C=4 for actions.")
    doc.add_paragraph("For matched neural comparisons, a word-only 128-dimensional SVD representation is followed by a word-plus-character 512-dimensional representation. The projection and standardization fit only training rows. A neural classifier supplies a frozen conditional mean for the diffusion model. Both heads use identical input features, thirty training epochs per head, seed 42 and development-based checkpoint selection. The feature follow-up changes both the representation and retained dimensionality; it does not isolate their individual effects.")
    doc.add_paragraph("The diffusion implementation adapts CARD [2]. Given a one-hot target y and frozen classifier prediction p(x), it diffuses the residual r0 = y − p(x). A network predicts injected Gaussian noise conditioned on text features, the noisy residual, p(x), and the timestep. A cosine schedule with 100 reverse steps and eight samples produces class votes. This generates labels, not conversation text. The architecture and schedule are an adaptation, not a reproduction of the paper's reported experiments. Sample agreement is not calibrated confidence.")
    semantic_ready=all((root/f"diffusion_{target}_semantic_seed42/test_metrics.json").exists() for target in ["risk","action"])
    doc.add_paragraph("A semantic-feature follow-up combines frozen 768-dimensional LaBSE embeddings [3] with 54 text/role measurements. The encoder uses a 256-token limit. Existing embeddings are reused and missing silver conversations are encoded; source text hashes and feature schemas are checked. Scaling fits training only. The matched heads use the same training protocol. " + ("Completed results appear in the tables below." if semantic_ready else "Its evaluation is pending in this document snapshot; no outcome is claimed."))
    doc.add_heading("4. Results",1)
    if (root/"majority_baselines.json").exists():
        majority=load(root/"majority_baselines.json")
        doc.add_paragraph(f"For scale, a constant predictor chosen only from the mixed training labels gives Kazakh risk macro-F1 {majority['risk']['test_kk']['macro_f1']:.4f} and supported-action macro-F1 {majority['action']['test_kk']['macro_f1_supported']:.4f}. A dedicated binary majority predictor gives {majority['dedicated_binary_majority']['test_kk']['macro_f1']:.4f}. These baselines contextualize learned scores; beating them does not establish reliability on new conversations.")
    for target,title in [("risk","4.1 Risk classification — three-class macro-F1"),("action","4.2 Action prediction — supported-class macro-F1")]:
        doc.add_heading(title,2)
        table=doc.add_table(rows=1,cols=3); table.style="Light Shading Accent 1"
        for c,v in zip(table.rows[0].cells,["Model","English","Kazakh"]): c.text=v
        for values in table_rows(target):
            for c,v in zip(table.add_row().cells,values): c.text=v
    doc.add_paragraph("The original mixed word baseline retains 0.8983 Kazakh merged-binary macro-F1. Risk sparse tuning changes three-class F1 only from 0.6675 to 0.6683 while reducing binary F1 to 0.8444; it is not an overall replacement for the original baseline. Action sparse tuning improves its supported-class point estimate from 0.4147 to 0.4328. These tasks and averaging conventions must remain explicit.")
    doc.add_paragraph("The richer lexical features raise diffusion risk F1 from 0.5131 to 0.5735 and MLP risk F1 from 0.5466 to 0.5898. For actions, diffusion rises from 0.3640 to 0.3872, while MLP decreases slightly from 0.3850 to 0.3798. These are all reported, rather than selecting only favorable changes.")
    doc.add_heading("5. Uncertainty and error analysis",1)
    doc.add_paragraph("Paired bootstrap intervals use 2,000 resamples of whole conversation families, preserving dependencies between restylings and alignment across models. For the hybrid512-versus-word128 diffusion risk gain, the 95% interval is approximately [−0.0043, +0.1289]; the positive point estimate does not establish a general improvement. Follow-up hypotheses were informed by earlier test results, and multiple exploratory comparisons further limit confirmatory claims.")
    if stability:
        doc.add_heading("5.1 Training-seed stability",2)
        doc.add_paragraph("The main result tables retain seed-42 point estimates. A separate fixed-protocol check repeats the semantic+hand MLP/diffusion models with seeds 0, 1 and 42 while holding cached features, data partitions and hyperparameters constant. All seeds are retained. The table below shows Kazakh mean and sample standard deviation; this is descriptive variability across three seeds, not a confidence interval or a fresh test set. Other architectures retain their original seed coverage.")
        table=doc.add_table(rows=1,cols=3); table.style="Light Shading Accent 1"
        for cell,text in zip(table.rows[0].cells,["Target / model","Mean macro-F1","Sample SD"]): cell.text=text
        for result in stability["summary"]:
            if result["language"]!="test_kk": continue
            for cell,text in zip(table.add_row().cells,[result["target"]+" / "+result["model"],f"{result['mean']:.4f}",f"{result['sample_std']:.4f}"]): cell.text=text
        doc.add_paragraph("A fixed equal-weight ensemble is also reported in the raw stability JSON: MLP averages class votes; diffusion averages saved sample-vote vectors. No weights or best seed are selected using test performance. It remains an exploratory analysis on the historical benchmark.")
    doc.add_paragraph("On the initial mixed word TF-IDF Kazakh run, 53 reference SCAM examples are predicted SUSPICIOUS and 16 are predicted LEGITIMATE. The initial compressed-feature neural models also miss attacks as LEGITIMATE. This motivates richer features but does not prove a single failure mechanism. The initial diffusion risk run fixes seven MLP errors and breaks thirteen previously correct examples; on actions it fixes seven and breaks fifteen. The saved predictions allow direct inspection.")
    doc.add_paragraph("Sixteen paired examples were exported for diagnostic review. A separate thirty-conversation action audit hides reference labels and model outputs, allows 'Cannot determine from text', and exports answers for scoring against the source. No human results have been collected in this snapshot. Agreement is not objective correctness: alternative actions may be defensible, and the source's full operational policy is not available to the reviewer. The stratified sample is diagnostic, not a population-quality estimate.")
    challenge=root/"challenge_evaluation/challenge_results.json"
    if challenge.exists():
        doc.add_heading("5.2 Emulated conversation challenge" if stability else "5.1 Emulated conversation challenge",2)
        doc.add_paragraph("A frozen set of 27 newly authored conversations covers nine base scenarios in English/Kazakh plus entity substitutions, readable typos, code-switching and long benign context before an attack. No complete example exactly duplicates the existing training/development/evaluation bundle. Expectations represent the author's intended interpretation and are not independently validated labels. Draft Kazakh wording requires review. Language pairs and variants are dependent; these cases are qualitative diagnostics, not a new accuracy benchmark.")
        doc.add_paragraph("The portable CPU demo runs both the original mixed word risk baseline and the development-selected sparse risk model, with the selected action classifier. Exported JSON/NPZ models reproduce the saved classifier predictions on 627 probe texts. It replays prefixes without future turns, displays risk/action predictions, and checks whether same-intent edits change them. Optional completed XLM-R inference uses exactly the same frozen conversations on CPU.")
        results={r["id"]:r for r in load(challenge)["results"]}
        table=doc.add_table(rows=1,cols=3); table.style="Light Shading Accent 1"
        for c,v in zip(table.rows[0].cells,["Authored case","Author-intended risk","Selected risk model"]): c.text=v
        for name in ["bank_code_request_kk","self_service_reset_kk","quoted_injection_lesson_en","gradual_escalation_kk"]:
            r=results[name]
            for c,v in zip(table.add_row().cells,[name,r["authored_expected_risk"],r["prediction"]["risk"]]): c.text=v
        doc.add_paragraph("The successful code-request example does not establish reliability. False positives on benign security discussions and a missed late malicious request expose failures under these authored scenarios. The cases are not rewritten after observing predictions, and they are not added to training. Source-benchmark scores should not be interpreted as proof of deployment performance.")
        reproduction=root/"challenge_evaluation/windows_reproduction.json"
        if reproduction.exists():
            evidence=load(reproduction)
            doc.add_paragraph(f"The supplied Windows replay was checked against the original local outputs. Its notebook metadata records Python {evidence['python_recorded']}; all {evidence['executed_code_cells']} code cells have recorded execution and no error outputs. All {evidence['matching_prediction_objects']} case prediction objects, including per-turn outputs, match the local replay exactly. Case data, model parameters and inference code hashes are unchanged. This supports cross-platform reproducibility on the same cases, not additional independent test coverage or improved model quality.")
    doc.add_heading("6. GPU experiment status",1)
    if gpu:
        r=gpu["run"]
        doc.add_paragraph(f"The imported full {r['model']} experiment [4] trained on mixed English and silver Kazakh with {r['epochs']} configured epochs, batch size {r['batch_size']}, accumulation {r['accumulation']}, {r['max_length']}-token inputs and {r['optimizer']}. Completion was checked against training logs; test IDs, family mappings and labels match the local bundle. Scores were recomputed from predictions. The risk result tables include the verified paired test values. These are single-seed results on the historical benchmark, not a production-safety estimate.")
        doc.add_paragraph("Training completed all three epochs (1,392 updates). The best development checkpoint was checkpoint-928 at epoch 2: macro-F1 0.9248 and accuracy 0.9278. Epoch 3 development macro-F1 fell to 0.9130, so final evaluation correctly restores epoch-2 weights. Replaying the supplied final model on CPU reproduces all 600 saved test predictions exactly.")
        doc.add_paragraph("XLM-R scores 0.6873 three-class / 0.9338 binary macro-F1 on English and 0.6341 / 0.8772 on Kazakh. The matched original mixed word baseline scores 0.6900 / 0.9492 and 0.6675 / 0.8983 respectively. The Kazakh three-class paired-family bootstrap interval for XLM-R minus TF-IDF is [−0.1025, +0.0332]; all four paired intervals include zero. This run establishes no improvement over the sparse baseline and does not establish that transformers generally perform worse.")
        doc.add_paragraph("Of 100 reference Kazakh SCAM cases, XLM-R predicts 25 SCAM, 62 SUSPICIOUS and 13 LEGITIMATE. Merging the attack classes therefore gives a different view from strict three-class scoring. The 256-token limit truncates 8,349/14,827 training inputs and 173/300 English and 185/300 Kazakh test inputs. Truncation is a limitation to investigate, not a demonstrated cause of the errors. The development-to-test gap exists in English too; language transfer alone cannot explain it. The 31 exact development/training duplicates across different family IDs remain a selection caveat.")
        replay=root/"challenge_evaluation_xlmr/challenge_results.json"
        if replay.exists():
            doc.add_paragraph("The completed XLM-R model was also replayed on every prefix of the same frozen 27 authored conversations. It matches 17 author-intended final risk labels (21 merged-binary), versus 14 (20 binary) for the original word baseline. These dependent, unreviewed diagnostic examples do not establish a benchmark improvement. XLM-R detects the authored late attacks in both languages, including long-context variants, but still flags benign quoted injection lessons and security-awareness posters as SCAM. Its action outputs remain those of the separate sparse action classifier; XLM-R action fine-tuning is not claimed.")
    else:
        doc.add_paragraph("The remote GTX 1650 Ti has passed smoke checks. The requested full experiment fine-tunes FacebookAI/xlm-roberta-base [4] on mixed English and silver Kazakh for three epochs, with batch size 1, accumulation 32, 256-token head/tail inputs and Adafactor. It saves recovery checkpoints and selects the best validation checkpoint. No completed GPU evaluation has been received. The report therefore makes no claim that fine-tuning improves the results.")
        if progress:
            doc.add_paragraph(f"The latest user-reported phase is {progress.get('latest_phase_report',progress['phase'])}. The most recently supplied metrics, from {progress.get('metric_report_phase',progress['phase'])}, are development macro-F1 {progress['development_macro_f1']:.4f}, accuracy {progress['development_accuracy']:.4f}, and merged-binary macro-F1 {progress['development_binary_macro_f1']:.4f}. These are pooled English-plus-silver development scores, not results on the Kazakh reference benchmark. They are excluded from the test result tables.")
    doc.add_heading("7. Limitations and next work",1)
    for text in [("Most architecture comparisons use seed 42; the semantic heads additionally have a three-seed stability check. A fresh evaluation set is needed after the historical benchmark has guided development." if stability else "Only one seed was run for the CPU comparisons; a fresh evaluation set is needed after the historical benchmark has guided development."),
                 "The risk taxonomy is derived from action annotations. Weak action/label identifiability is a hypothesis that needs independent human judgment.",
                 "Kazakh references lack completed native-speaker review. Machine-translated development wording can differ from reference evaluation wording.",
                 "The action benchmark supports five of thirteen actions, including only one ignore example; supported-class macro-F1 is sensitive to this coverage.",
                 "Frozen encoders truncate input, while lexical features see the whole flattened conversation. The semantic combination changes multiple feature properties; attribution requires further ablations.",
                 "Next priorities are blinded label/translation review, removal of exact development duplicates, fixed-seed replication and context-length ablations, and a broader reviewed family-safe benchmark."]:
        doc.add_paragraph(text,style="List Bullet")
    doc.add_heading("8. Reproducibility",1)
    doc.add_paragraph("Scripts and artifacts are in the repository. train_transformer.py runs risk/action fine-tuning and sparse baselines; run_label_diffusion.py trains and evaluates matched dense models; tune_text_models.py performs development-only sparse selection; prepare_semantic_features.py creates frozen semantic/text caches; summarize_deadline_results.py and summarize_improvements.py assemble results and family-bootstrap intervals. The results snapshot is data/experiments/deadline_2026-11-23/. Model state and feature caches remain local; submitted metrics/predictions are compact exports.")
    doc.add_heading("References",1)
    for reference in references: doc.add_paragraph(reference)
    doc.save(out/"ScamBench_KK_report.docx")

    # PowerPoint: native editable text, tables and charts.
    presentation=Presentation(); presentation.slide_width=Inches(13.333); presentation.slide_height=Inches(7.5)
    blue=RGBColor.from_string("184F82"); gray=RGBColor.from_string("536273")
    def textbox(slide,text,x,y,w,h,size=20,bold=False,color=None):
        shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
        tf=shape.text_frame; tf.word_wrap=True
        for i,line in enumerate(text.split("\n")):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.text=line
            p.font.size=Pt(size); p.font.bold=bold; p.font.name="Aptos"
            if color: p.font.color.rgb=color
        return shape
    def slide(title,notes=""):
        s=presentation.slides.add_slide(presentation.slide_layouts[6])
        textbox(s,title,.55,.35,12.2,.85,28,True,blue)
        textbox(s,"ScamBench KK | 23 November 2026 | "+gpu_status,.55,7.05,11,.25,10,color=gray)
        textbox(s,str(len(presentation.slides)),12.2,7.05,.5,.25,10,color=gray)
        if notes and s.notes_slide.notes_text_frame is not None: s.notes_slide.notes_text_frame.text=notes
        return s
    def bullets(s,items):
        y=1.55
        for text in items:
            textbox(s,"• "+text,.7,y,11.8,1.05,21)
            y+=1.18
    def native_table(s,header,values,y=1.6):
        table=s.shapes.add_table(len(values)+1,len(header),Inches(.65),Inches(y),Inches(12),Inches(min(4.8,.43*(len(values)+1)))).table
        widths=[7,2.5,2.5] if len(header)==3 else [12/len(header)]*len(header)
        for col,width in zip(table.columns,widths): col.width=Inches(width)
        for i,values in enumerate([header]+values):
            for j,value in enumerate(values):
                cell=table.cell(i,j); cell.text=str(value)
                for p in cell.text_frame.paragraphs:
                    p.font.size=Pt(14); p.font.bold=(i==0)
                if i==0: cell.fill.solid(); cell.fill.fore_color.rgb=RGBColor.from_string("E9F0F7")
    s=slide("ScamBench-KK: dataset, decisions and diffusion")
    bullets(s,["A Kazakh extension and three-class risk taxonomy.","Family-safe evaluation and English-to-Kazakh transfer.","Offline action prediction and conditional label diffusion.","Completed CPU results; "+gpu_status+"."])
    s=slide("Questions and contributions")
    bullets(s,["Can models trained on English recognize the same conversations in Kazakh?","Does silver Kazakh training improve transfer?","Can text predict the agent's recorded action?","Does a conditional diffusion head improve on a matched neural classifier?"])
    s=slide("Dataset and targets")
    bullets(s,["Local processed source: 37,421 conversations; merged extension: 37,721 rows.","Kazakh reference slice: 300 conversations, 100 per risk class; native review pending.","Risk labels: LEGITIMATE / SUSPICIOUS / SCAM. Binary reporting merges the last two.","Action target: 13 recorded decisions; only 5 occur in the current test set."])
    s=slide("Preventing contamination")
    bullets(s,["Restylings of the same conversation cross the original splits: 83% validation overlap in the project audit.","Current experiments keep families and their translations together.","Train: 11,227 English + 3,600 silver. Dev: 1,208 English + 399 silver.","Evaluate: 300 EN/KK pairs, 273 families. The benchmark is historical, not a fresh confirmatory test."])
    if duplicate_audit.exists():
        s=slide("Additional audit: exact text can cross family IDs")
        audit=load(duplicate_audit)["exact"]
        bullets(s,[f"Development: {audit['development']['overlapping_rows']} / {audit['development']['rows']} rows exactly duplicate training text despite disjoint family IDs.",
            f"Paired test: {audit['test']['overlapping_rows']} / {audit['test']['rows']} exact training duplicates.",
            "Potential bias in development-based selection; no retrospective change to the active run.",
            "Next split version should group identical texts across IDs as well as declared families. This check does not exclude near-duplicates."])
    for target,title in [("risk","Risk classification — three-class macro-F1"),("action","Action prediction — five supported classes")]:
        s=slide(title)
        native_table(s,["Model","English","Kazakh"],table_rows(target))
        note="Action support: refuse 99, engage 88, verify 87, audit 25, ignore 1. Trained schema: 13." if target=="action" else "Merged-binary Kazakh score of the original mixed word baseline: 0.8983. Different target."
        textbox(s,note,.7,6.45,11.8,.45,13,color=gray)
        if target=="risk":
            s=slide("Cross-lingual risk results — completed CPU models")
            names=[("Word TF-IDF, English","TF-IDF EN"),("Word TF-IDF, mixed","TF-IDF mixed"),
                   ("Hybrid512: mlp","MLP hybrid"),("Hybrid512: diffusion","Diffusion hybrid")]
            if semantic_ready:
                names += [("LaBSE + 54 text features: mlp","MLP semantic"),
                          ("LaBSE + 54 text features: diffusion","Diffusion semantic")]
            if gpu: names += [("XLM-R mixed (GPU verified)","XLM-R mixed")]
            lookup={(r["model"],r["language"]):r["metrics"]["macro_f1"] for r in rows if r["target"]=="risk"}
            chart_data=CategoryChartData(); chart_data.categories=[short for _,short in names]
            for lang,label in [("test_en","English"),("test_kk","Kazakh")]:
                chart_data.add_series(label,[lookup[name,lang] for name,_ in names])
            chart=s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,Inches(.7),Inches(1.6),Inches(11.9),Inches(4.9),chart_data).chart
            chart.has_legend=True; chart.legend.position=XL_LEGEND_POSITION.BOTTOM
            chart.value_axis.maximum_scale=1.0; chart.value_axis.minimum_scale=0.0
            chart.category_axis.tick_labels.font.size=Pt(12)
            chart.plots[0].has_data_labels=True; chart.plots[0].data_labels.number_format="0.000"
            chart.plots[0].data_labels.font.size=Pt(11)
    s=slide("Diffusion model: generate labels conditioned on text")
    bullets(s,["CARD-inspired adaptation: frozen classifier supplies a conditional mean p(x).","Add Gaussian noise to the residual of the one-hot label: y − p(x).","A network predicts noise from text features, timestep, noisy label and p(x).","Reverse denoising: 100 steps, 8 samples; class votes are not calibrated safety confidence."])
    s=slide("Controlled feature follow-up")
    bullets(s,["Initial representation: word TF-IDF compressed to 128 components.","Follow-up: word + character TF-IDF retaining 512 components.","Both MLP and diffusion share features, split, seed and checkpoint rule; 30 epochs per head.","Risk point estimates improve, but action MLP regresses slightly. Report both outcomes."])
    s=slide("Semantic-feature follow-up")
    bullets(s,["Frozen LaBSE: 768 semantic dimensions with a 256-token encoder limit.","Append 54 text/role measurements; annotations and recorded actions remain excluded.","Reuse existing embeddings, encode missing silver rows, verify text hashes and fit scaling on train only.","Status: "+("completed; included in result tables." if semantic_ready else "evaluation pending; no outcome claimed.")])
    s=slide("Error analysis and human review")
    bullets(s,["Initial word TF-IDF: 53 SCAM examples become SUSPICIOUS; 16 become LEGITIMATE.","The weaker compressed-feature models also miss attacks as legitimate; a single cause is not established.","16 paired model examples are ready for inspection; no review conclusions have been invented.","New blind action audit: 30 conversations, hidden references and an explicit 'Cannot determine' option."])
    s=slide("Interpretation and limitations")
    bullets(s,[("Most architecture comparisons are single-seed; semantic-head stability is checked across three seeds." if stability else "Single-seed exploratory results on a reused historical benchmark."),"Action agreement measures source-label prediction, not execution success or safety.","Family-bootstrap intervals preserve dependence between restylings; small gains are uncertain.","Native-speaker review and broader action coverage are outstanding."])
    if stability:
        s=slide("Semantic-head stability: three fixed seeds")
        values=[]
        for result in stability["summary"]:
            if result["language"]=="test_kk": values.append([result["target"]+" / "+result["model"],f"{result['mean']:.4f}",f"{result['sample_std']:.4f}"])
        native_table(s,["Target / model","Mean macro-F1","Sample SD"],values,y=1.8)
        textbox(s,"Seeds 0, 1, 42; fixed cached features, split and hyperparameters. Every seed is reported.",.7,4.8,11.8,.8,18)
        textbox(s,"Three-seed SD is descriptive variability, not a confidence interval. Other model families remain single-seed.",.7,5.7,11.8,.8,18)
    if challenge.exists():
        results={r["id"]:r for r in load(challenge)["results"]}
        s=slide("Emulated conversations: successes and failures")
        values=[]
        for name in ["bank_code_request_kk","self_service_reset_kk","quoted_injection_lesson_en","gradual_escalation_kk"]:
            r=results[name]; values.append([name,r["authored_expected_risk"],r["prediction"]["risk"]])
        native_table(s,["Scenario","Author-intended risk","Model risk"],values,y=1.8)
        textbox(s,"27 authored cases; nine base scenarios plus variants. Not a new gold benchmark.",.7,4.65,11.8,.6,18)
        textbox(s,"The notebook replays turns and supports your own conversations. All failures are retained.",.7,5.45,11.8,.7,18)
        if (root/"challenge_evaluation/windows_reproduction.json").exists():
            textbox(s,"Supplied Windows replay: 27/27 prediction objects match the local run. Same cases, no new benchmark.",.7,6.35,11.8,.4,13,color=gray)
    if gpu:
        s=slide("XLM-R completed: high development score, weaker transfer",
                "Three epochs completed; final weights restore the best epoch-2 checkpoint. All 600 supplied test predictions were reproduced locally. This is one configuration, not a general verdict on transformers.")
        bullets(s,["Best checkpoint: epoch 2, update 928; development macro-F1 0.9248.",
                   "Kazakh: XLM-R 0.6341 / 0.8772; word TF-IDF 0.6675 / 0.8983 (three-class / binary macro-F1).",
                   "Of 100 Kazakh SCAM cases: 25 SCAM, 62 SUSPICIOUS, 13 LEGITIMATE. Paired intervals show no established gain.",
                   "Same frozen demo: catches late attacks; still flags benign security lessons. Context truncation and label review need investigation."])
    s=slide("Next steps")
    bullets(s,[("Inspect verified XLM-R per-class errors and paired-family comparisons." if gpu else "Collect the three-epoch XLM-R mixed-data evaluation; do not claim its outcome early."),"Complete independent label/translation review and examine disagreements.","Repeat fixed settings across seeds and evaluate on a fresh, broader family-safe test.","Retain negative results and accuracy/binary-score trade-offs alongside improvements."])
    if progress and not gpu:
        s=slide("XLM-R progress — development scores only")
        bullets(s,["User-reported phase: "+progress.get("latest_phase_report",progress["phase"])+".",
            f"Last supplied development metrics: F1 {progress['development_macro_f1']:.4f}; accuracy {progress['development_accuracy']:.4f}; binary F1 {progress['development_binary_macro_f1']:.4f} (earlier report).",
            "Development mixes 1,208 English and 399 machine-translated Kazakh rows. This is not the 300-pair reference test.",
            "Finish the planned run, select the best validation checkpoint, then evaluate paired EN/KK. Final test results remain pending."])
    s=slide("References")
    for i,reference in enumerate(references): textbox(s,reference,.7,1.5+i*1.2,11.8,1,15)
    presentation.save(out/"ScamBench_KK_presentation.pptx")
    qa=pathlib.Path(__file__).resolve().parents[1]/"reports/DEFENSE_QA.txt"
    if qa.exists():
        (out/"defense_questions.txt").write_text(qa.read_text(encoding="utf-8"),encoding="utf-8")
    with zipfile.ZipFile(out/"submission_pack.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p.suffix!=".zip": z.write(p,p.name)
        for p in [root/"defense/defense_slides.pdf",root/"defense/speaking_notes.txt"]:
            z.write(p,p.name)
        if gpu:
            for p in (root/"xlmr_mixed_risk").glob("*.json"):
                z.write(p,"xlmr_verified/"+p.name)
        replay=root/"challenge_evaluation_xlmr"
        if replay.exists():
            for p in replay.iterdir():
                if p.is_file(): z.write(p,"xlmr_conversation_replay/"+p.name)
        audit=pathlib.Path("review/action_audit/blind_action_audit_30.html")
        if audit.exists(): z.write(audit,audit.name)
        card=pathlib.Path("data/release/README.md")
        if card.exists(): z.write(card,"DATASET_CARD.md")
        if duplicate_audit.exists(): z.write(duplicate_audit,duplicate_audit.name)
        reproduction=root/"challenge_evaluation/windows_reproduction.json"
        if reproduction.exists(): z.write(reproduction,reproduction.name)
        if stability_path.exists():
            z.write(stability_path,"seed_stability/summary.json")
            z.write(root/"seed_stability/per_seed.csv","seed_stability/per_seed.csv")
    print("Editable Word report:",out/"ScamBench_KK_report.docx")
    print("Editable PowerPoint:",out/"ScamBench_KK_presentation.pptx")
    print("Pack:",out/"submission_pack.zip")


if __name__=="__main__": main()
