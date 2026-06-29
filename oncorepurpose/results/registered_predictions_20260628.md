# OncoEvidence forward prediction set (registered)

_2026-06-28T23:32:02.495845+00:00_

- **git commit:** `54297932d08f01ec7b77996595e39efcc29dad4a`
- **prediction sha256:** `d7bb30423e808f72912e40fa99c7f1bcfd07d5e95dbdc15cd03f76133358b2fd`
- **predictions:** 60  ·  mode: full  ·  device: cuda

> Timestamped, hash-committed NOVEL drug→cancer predictions for prospective checking against future ClinicalTrials.gov entries / approvals. Novelty: no existing indication/contraindication/off-label edge (exclude_known=True) AND a non-empty graph MOA path (direct target / PPI / shared pathway). Hypothesis-generating; not medical advice.

## Predictions

| Cancer | Drug | Support | Model score | Lift | Top MOA path |
|---|---|---|---|---|---|
| B-cell neoplasm | Capsaicin | direct-target mechanism | 0.881 | +0.763 | Capsaicin --targets--> CYP2E1 <--associated-- B-cell neoplasm |
| B-cell neoplasm | Fluphenazine | direct-target mechanism | 0.879 | +0.763 | Fluphenazine --targets--> CYP2E1 <--associated-- B-cell neoplasm |
| B-cell neoplasm | Naloxone | interaction-level mechanism | 0.877 | +0.763 | Naloxone --targets--> CREB1 --interacts--> GSK3B <--associated-- B-cell neoplasm |
| B-cell neoplasm | Plerixafor | interaction-level mechanism | 0.877 | +0.763 | Plerixafor --targets--> CXCR4 --interacts--> CD79B <--associated-- B-cell neoplasm |
| B-cell neoplasm | Tolmetin | interaction-level mechanism | 0.875 | +0.763 | Tolmetin --targets--> TDO2 --interacts--> EIF4E <--associated-- B-cell neoplasm |
| Hodgkins lymphoma | Acetaminophen | direct-target mechanism | 0.944 | +0.835 | Acetaminophen --targets--> UGT1A1 <--associated-- Hodgkins lymphoma |
| Hodgkins lymphoma | Clomipramine | direct-target mechanism | 0.944 | +0.836 | Clomipramine --targets--> GSTP1 <--associated-- Hodgkins lymphoma |
| Hodgkins lymphoma | Isoprenaline | interaction-level mechanism | 0.946 | +0.835 | Isoprenaline --targets--> PIK3R1 --interacts--> SOCS1 <--associated-- Hodgkins lymphoma |
| Hodgkins lymphoma | Nadroparin | interaction-level mechanism | 0.944 | +0.835 | Nadroparin --targets--> MYC --interacts--> MTHFD2 <--associated-- Hodgkins lymphoma |
| Hodgkins lymphoma | Naltrexone | direct-target mechanism | 0.944 | +0.837 | Naltrexone --targets--> UGT1A1 <--associated-- Hodgkins lymphoma |
| acute lymphoblastic/lymphocytic leukemia | Folic acid | direct-target mechanism | 0.869 | +0.744 | Folic acid --targets--> ABCG2 <--associated-- acute lymphoblastic/lymphocytic leukemia |
| acute lymphoblastic/lymphocytic leukemia | Hydroxocobalamin | interaction-level mechanism | 0.861 | +0.743 | Hydroxocobalamin --targets--> MMACHC --interacts--> IRF4 <--associated-- acute lymphoblastic/lymphocytic leukemia |
| acute lymphoblastic/lymphocytic leukemia | Ivermectin | direct-target mechanism | 0.870 | +0.743 | Ivermectin --targets--> ABCG2 <--associated-- acute lymphoblastic/lymphocytic leukemia |
| acute lymphoblastic/lymphocytic leukemia | Pramlintide | interaction-level mechanism | 0.860 | +0.743 | Pramlintide --targets--> CALCR --interacts--> TOPBP1 <--associated-- acute lymphoblastic/lymphocytic leukemia |
| acute lymphoblastic/lymphocytic leukemia | Salmon calcitonin | interaction-level mechanism | 0.864 | +0.744 | Salmon calcitonin --targets--> CALCR --interacts--> TOPBP1 <--associated-- acute lymphoblastic/lymphocytic leukemia |
| classic Hodgkin lymphoma | Capsaicin | interaction-level mechanism | 0.858 | +0.740 | Capsaicin --targets--> PHB2 --interacts--> CDK2 <--associated-- classic Hodgkin lymphoma |
| classic Hodgkin lymphoma | Insulin glargine | interaction-level mechanism | 0.864 | +0.741 | Insulin glargine --targets--> INSR --interacts--> SOCS1 <--associated-- classic Hodgkin lymphoma |
| classic Hodgkin lymphoma | Insulin human | interaction-level mechanism | 0.860 | +0.740 | Insulin human --targets--> RB1 --interacts--> CDK2 <--associated-- classic Hodgkin lymphoma |
| classic Hodgkin lymphoma | Insulin pork | interaction-level mechanism | 0.865 | +0.740 | Insulin pork --targets--> CCN3 --interacts--> CDK2 <--associated-- classic Hodgkin lymphoma |
| classic Hodgkin lymphoma | Simvastatin | direct-target mechanism | 0.863 | +0.740 | Simvastatin --targets--> UGT1A1 <--associated-- classic Hodgkin lymphoma |
| leukemia, lymphocytic, susceptibility to | Capsaicin | direct-target mechanism | 0.858 | +0.741 | Capsaicin --targets--> CYP2E1 <--associated-- leukemia, lymphocytic, susceptibility to |
| leukemia, lymphocytic, susceptibility to | Fluticasone furoate | direct-target mechanism | 0.870 | +0.741 | Fluticasone furoate --targets--> CYP2C8 <--associated-- leukemia, lymphocytic, susceptibility to |
| leukemia, lymphocytic, susceptibility to | Insulin human | direct-target mechanism | 0.861 | +0.741 | Insulin human --targets--> RB1 <--associated-- leukemia, lymphocytic, susceptibility to |
| leukemia, lymphocytic, susceptibility to | Insulin pork | direct-target mechanism | 0.865 | +0.741 | Insulin pork --targets--> RB1 <--associated-- leukemia, lymphocytic, susceptibility to |
| leukemia, lymphocytic, susceptibility to | Zopiclone | direct-target mechanism | 0.862 | +0.741 | Zopiclone --targets--> CYP2C8 <--associated-- leukemia, lymphocytic, susceptibility to |
| lung cancer | Hyaluronidase | direct-target mechanism | 0.861 | +0.744 | Hyaluronidase --targets--> TGFB1 <--associated-- lung cancer |
| lung cancer | Ivermectin | direct-target mechanism | 0.869 | +0.743 | Ivermectin --targets--> SLCO1B3 <--associated-- lung cancer |
| lung cancer | Mivacurium | direct-target mechanism | 0.865 | +0.743 | Mivacurium --targets--> CHRNA2 <--associated-- lung cancer |
| lung cancer | Pramlintide | direct-target mechanism | 0.859 | +0.743 | Pramlintide --targets--> RAMP2 <--associated-- lung cancer |
| lung cancer | Varenicline | direct-target mechanism | 0.862 | +0.742 | Varenicline --targets--> CHRNA3 <--associated-- lung cancer |
| lymphosarcoma | Capsaicin | direct-target mechanism | 0.885 | +0.767 | Capsaicin --targets--> CYP2E1 <--associated-- lymphosarcoma |
| lymphosarcoma | Chlorpromazine | direct-target mechanism | 0.885 | +0.767 | Chlorpromazine --targets--> CYP2E1 <--associated-- lymphosarcoma |
| lymphosarcoma | Citric acid | interaction-level mechanism | 0.882 | +0.767 | Citric acid --targets--> HGS --interacts--> SYK <--associated-- lymphosarcoma |
| lymphosarcoma | Naloxone | interaction-level mechanism | 0.881 | +0.767 | Naloxone --targets--> CREB1 --interacts--> GSK3B <--associated-- lymphosarcoma |
| lymphosarcoma | Tirofiban | interaction-level mechanism | 0.885 | +0.767 | Tirofiban --targets--> ITGB3 --interacts--> SYK <--associated-- lymphosarcoma |
| mantle cell lymphoma | Bromazepam | direct-target mechanism | 0.854 | +0.732 | Bromazepam --targets--> CYP2E1 <--associated-- mantle cell lymphoma |
| mantle cell lymphoma | Capsaicin | direct-target mechanism | 0.850 | +0.732 | Capsaicin --targets--> CYP2E1 <--associated-- mantle cell lymphoma |
| mantle cell lymphoma | Ferrous ascorbate | direct-target mechanism | 0.855 | +0.732 | Ferrous ascorbate --targets--> TFRC <--associated-- mantle cell lymphoma |
| mantle cell lymphoma | Trabectedin | direct-target mechanism | 0.857 | +0.732 | Trabectedin --targets--> CYP2E1 <--associated-- mantle cell lymphoma |
| mantle cell lymphoma | Zopiclone | direct-target mechanism | 0.854 | +0.733 | Zopiclone --targets--> CYP2E1 <--associated-- mantle cell lymphoma |
| non-small cell lung carcinoma (disease) | Amobarbital | direct-target mechanism | 0.859 | +0.739 | Amobarbital --targets--> GRIK2 <--associated-- non-small cell lung carcinoma (disease) |
| non-small cell lung carcinoma (disease) | Capsaicin | direct-target mechanism | 0.856 | +0.738 | Capsaicin --targets--> CYP2E1 <--associated-- non-small cell lung carcinoma (disease) |
| non-small cell lung carcinoma (disease) | Insulin pork | direct-target mechanism | 0.863 | +0.738 | Insulin pork --targets--> RB1 <--associated-- non-small cell lung carcinoma (disease) |
| non-small cell lung carcinoma (disease) | Pentobarbital | direct-target mechanism | 0.861 | +0.739 | Pentobarbital --targets--> GRIK2 <--associated-- non-small cell lung carcinoma (disease) |
| non-small cell lung carcinoma (disease) | Zopiclone | direct-target mechanism | 0.859 | +0.738 | Zopiclone --targets--> CYP2E1 <--associated-- non-small cell lung carcinoma (disease) |
| precursor T-cell acute lymphoblastic leukemia | Desmopressin | interaction-level mechanism | 0.889 | +0.774 | Desmopressin --targets--> AVPR2 --interacts--> TAL1 <--associated-- precursor T-cell acute lymphoblastic leukemia |
| precursor T-cell acute lymphoblastic leukemia | Hyaluronidase | interaction-level mechanism | 0.891 | +0.775 | Hyaluronidase --targets--> TGFB1 --interacts--> CTCF <--associated-- precursor T-cell acute lymphoblastic leukemia |
| precursor T-cell acute lymphoblastic leukemia | Lindane | interaction-level mechanism | 0.892 | +0.774 | Lindane --targets--> GABRR1 --interacts--> BCR <--associated-- precursor T-cell acute lymphoblastic leukemia |
| precursor T-cell acute lymphoblastic leukemia | Naloxone | interaction-level mechanism | 0.887 | +0.774 | Naloxone --targets--> CREB1 --interacts--> CCND3 <--associated-- precursor T-cell acute lymphoblastic leukemia |
| precursor T-cell acute lymphoblastic leukemia | Tromethamine | interaction-level mechanism | 0.887 | +0.775 | Tromethamine --targets--> APP --interacts--> SET <--associated-- precursor T-cell acute lymphoblastic leukemia |
| primary cutaneous T-cell lymphoma | Citric acid | interaction-level mechanism | 0.883 | +0.768 | Citric acid --targets--> C8G --interacts--> CTCF <--associated-- primary cutaneous T-cell lymphoma |
| primary cutaneous T-cell lymphoma | Heparin | interaction-level mechanism | 0.886 | +0.768 | Heparin --targets--> FGFR1 --interacts--> RPS6KA1 <--associated-- primary cutaneous T-cell lymphoma |
| primary cutaneous T-cell lymphoma | Hyaluronidase | interaction-level mechanism | 0.885 | +0.769 | Hyaluronidase --targets--> TGFB1 --interacts--> CTCF <--associated-- primary cutaneous T-cell lymphoma |
| primary cutaneous T-cell lymphoma | Hydroxocobalamin | interaction-level mechanism | 0.886 | +0.768 | Hydroxocobalamin --targets--> MMACHC --interacts--> IRF4 <--associated-- primary cutaneous T-cell lymphoma |
| primary cutaneous T-cell lymphoma | Naloxone | interaction-level mechanism | 0.882 | +0.768 | Naloxone --targets--> CREB1 --interacts--> PRKG1 <--associated-- primary cutaneous T-cell lymphoma |
| primary cutaneous T-cell non-Hodgkin lymphoma | Carboprost tromethamine | interaction-level mechanism | 0.889 | +0.769 | Carboprost tromethamine --targets--> PTGER1 --interacts--> CTCF <--associated-- primary cutaneous T-cell non-Hodgkin lymphoma |
| primary cutaneous T-cell non-Hodgkin lymphoma | Hyaluronidase | interaction-level mechanism | 0.887 | +0.771 | Hyaluronidase --targets--> TGFB1 --interacts--> CTCF <--associated-- primary cutaneous T-cell non-Hodgkin lymphoma |
| primary cutaneous T-cell non-Hodgkin lymphoma | Hydroxocobalamin | interaction-level mechanism | 0.888 | +0.769 | Hydroxocobalamin --targets--> MMACHC --interacts--> IRF4 <--associated-- primary cutaneous T-cell non-Hodgkin lymphoma |
| primary cutaneous T-cell non-Hodgkin lymphoma | Naloxone | interaction-level mechanism | 0.883 | +0.769 | Naloxone --targets--> TLR4 --interacts--> IRF4 <--associated-- primary cutaneous T-cell non-Hodgkin lymphoma |
| primary cutaneous T-cell non-Hodgkin lymphoma | Tromethamine | interaction-level mechanism | 0.882 | +0.770 | Tromethamine --targets--> APP --interacts--> CTCF <--associated-- primary cutaneous T-cell non-Hodgkin lymphoma |

## How to check this later

Re-hash the sorted `predictions` (drug, cancer, top MOA path text) and confirm it matches `prediction_sha256`; the git commit pins the exact code that produced them. Then query ClinicalTrials.gov / approvals for each (drug, cancer) pair dated AFTER this timestamp.
