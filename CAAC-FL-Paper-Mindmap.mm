<map version="1.0.1">
<!-- To view this file, download free mind mapping software FreeMind from http://freemind.sourceforge.net -->
<node CREATED="1700000000000" ID="ID_ROOT" MODIFIED="1700000000000" TEXT="CAAC-FL: Client-Adaptive Anomaly-Aware Clipping for Healthcare FL">
<node CREATED="1700000000000" ID="ID_1" MODIFIED="1700000000000" POSITION="right" TEXT="1. Problem Context">
<node CREATED="1700000000000" ID="ID_1_1" MODIFIED="1700000000000" TEXT="Healthcare Federated Learning">
<node CREATED="1700000000000" ID="ID_1_1_1" MODIFIED="1700000000000" TEXT="Enable collaborative training without data sharing"/>
<node CREATED="1700000000000" ID="ID_1_1_2" MODIFIED="1700000000000" TEXT="HIPAA and GDPR compliance"/>
<node CREATED="1700000000000" ID="ID_1_1_3" MODIFIED="1700000000000" TEXT="Critical: model errors impact patient outcomes"/>
</node>
<node CREATED="1700000000000" ID="ID_1_2" MODIFIED="1700000000000" TEXT="Byzantine Threat">
<node CREATED="1700000000000" ID="ID_1_2_1" MODIFIED="1700000000000" TEXT="Compromised/faulty participants"/>
<node CREATED="1700000000000" ID="ID_1_2_2" MODIFIED="1700000000000" TEXT="Corrupted gradient updates"/>
<node CREATED="1700000000000" ID="ID_1_2_3" MODIFIED="1700000000000" TEXT="Can catastrophically degrade performance"/>
<node CREATED="1700000000000" ID="ID_1_2_4" MODIFIED="1700000000000" TEXT="Single Byzantine node can increase misdiagnosis"/>
</node>
<node CREATED="1700000000000" ID="ID_1_3" MODIFIED="1700000000000" TEXT="Key Challenge">
<node CREATED="1700000000000" ID="ID_1_3_1" MODIFIED="1700000000000" TEXT="Data Heterogeneity in Healthcare">
<node CREATED="1700000000000" ID="ID_1_3_1_1" MODIFIED="1700000000000" TEXT="Pediatric vs Geriatric hospitals"/>
<node CREATED="1700000000000" ID="ID_1_3_1_2" MODIFIED="1700000000000" TEXT="Specialized medical centers"/>
<node CREATED="1700000000000" ID="ID_1_3_1_3" MODIFIED="1700000000000" TEXT="Natural statistical diversity"/>
</node>
<node CREATED="1700000000000" ID="ID_1_3_2" MODIFIED="1700000000000" TEXT="Distinguishing legitimate diversity from attacks">
<node CREATED="1700000000000" ID="ID_1_3_2_1" MODIFIED="1700000000000" TEXT="Current methods: uniform filtering rules"/>
<node CREATED="1700000000000" ID="ID_1_3_2_2" MODIFIED="1700000000000" TEXT="Result: &lt;10% accuracy in heterogeneous settings"/>
<node CREATED="1700000000000" ID="ID_1_3_2_3" MODIFIED="1700000000000" TEXT="Valuable contributions excluded"/>
</node>
</node>
</node>
<node CREATED="1700000000000" ID="ID_2" MODIFIED="1700000000000" POSITION="right" TEXT="2. Related Work Limitations">
<node CREATED="1700000000000" ID="ID_2_1" MODIFIED="1700000000000" TEXT="Statistical Filtering">
<node CREATED="1700000000000" ID="ID_2_1_1" MODIFIED="1700000000000" TEXT="Trimmed Mean (Yin et al., 2018)">
<node CREATED="1700000000000" ID="ID_2_1_1_1" MODIFIED="1700000000000" TEXT="Coordinate-wise outlier removal"/>
<node CREATED="1700000000000" ID="ID_2_1_1_2" MODIFIED="1700000000000" TEXT="Fails when attacks normal in most dimensions"/>
</node>
</node>
<node CREATED="1700000000000" ID="ID_2_2" MODIFIED="1700000000000" TEXT="Geometric Methods">
<node CREATED="1700000000000" ID="ID_2_2_1" MODIFIED="1700000000000" TEXT="Krum (Blanchard et al., 2017)">
<node CREATED="1700000000000" ID="ID_2_2_1_1" MODIFIED="1700000000000" TEXT="Selects central updates by distance"/>
<node CREATED="1700000000000" ID="ID_2_2_1_2" MODIFIED="1700000000000" TEXT="Filters legitimate diverse contributions"/>
</node>
</node>
<node CREATED="1700000000000" ID="ID_2_3" MODIFIED="1700000000000" TEXT="Adaptive Methods">
<node CREATED="1700000000000" ID="ID_2_3_1" MODIFIED="1700000000000" TEXT="ARC (Baruch et al., 2019)"/>
<node CREATED="1700000000000" ID="ID_2_3_2" MODIFIED="1700000000000" TEXT="RFA (Pillutla et al., 2021)"/>
<node CREATED="1700000000000" ID="ID_2_3_3" MODIFIED="1700000000000" TEXT="Still apply global rules"/>
<node CREATED="1700000000000" ID="ID_2_3_4" MODIFIED="1700000000000" TEXT="Cannot accommodate institutional differences"/>
</node>
<node CREATED="1700000000000" ID="ID_2_4" MODIFIED="1700000000000" TEXT="Trust-Based">
<node CREATED="1700000000000" ID="ID_2_4_1" MODIFIED="1700000000000" TEXT="FLTrust (Cao et al., 2021)">
<node CREATED="1700000000000" ID="ID_2_4_1_1" MODIFIED="1700000000000" TEXT="Uses server-side root dataset"/>
<node CREATED="1700000000000" ID="ID_2_4_1_2" MODIFIED="1700000000000" TEXT="Violates privacy principles"/>
</node>
</node>
<node CREATED="1700000000000" ID="ID_2_5" MODIFIED="1700000000000" TEXT="Recent Approaches">
<node CREATED="1700000000000" ID="ID_2_5_1" MODIFIED="1700000000000" TEXT="LASA (Xu et al., 2024)">
<node CREATED="1700000000000" ID="ID_2_5_1_1" MODIFIED="1700000000000" TEXT="Layer-wise sparsification"/>
</node>
<node CREATED="1700000000000" ID="ID_2_5_2" MODIFIED="1700000000000" TEXT="BR-MTRL (Le &amp; Moothedath, 2025)">
<node CREATED="1700000000000" ID="ID_2_5_2_1" MODIFIED="1700000000000" TEXT="Client-specific layers"/>
<node CREATED="1700000000000" ID="ID_2_5_2_2" MODIFIED="1700000000000" TEXT="Lacks temporal tracking"/>
</node>
<node CREATED="1700000000000" ID="ID_2_5_3" MODIFIED="1700000000000" TEXT="Clipped Clustering (Li et al., 2024)">
<node CREATED="1700000000000" ID="ID_2_5_3_1" MODIFIED="1700000000000" TEXT="Demonstrated &lt;10% accuracy in non-IID"/>
</node>
</node>
<node CREATED="1700000000000" ID="ID_2_6" MODIFIED="1700000000000" TEXT="Fundamental Limitation">
<node CREATED="1700000000000" ID="ID_2_6_1" MODIFIED="1700000000000" TEXT="Treat all participants as statistically identical"/>
<node CREATED="1700000000000" ID="ID_2_6_2" MODIFIED="1700000000000" TEXT="Global thresholds inadequate"/>
<node CREATED="1700000000000" ID="ID_2_6_3" MODIFIED="1700000000000" TEXT="Gap: theory exists (Werner et al., 2023) but practical implementations nascent"/>
</node>
</node>
<node CREATED="1700000000000" ID="ID_3" MODIFIED="1700000000000" POSITION="right" TEXT="3. CAAC-FL Approach">
<node CREATED="1700000000000" ID="ID_3_1" MODIFIED="1700000000000" TEXT="Core Innovation">
<node CREATED="1700000000000" ID="ID_3_1_1" MODIFIED="1700000000000" TEXT="Client-Specific Behavioral Profiling">
<node CREATED="1700000000000" ID="ID_3_1_1_1" MODIFIED="1700000000000" TEXT="Node-level monitoring vs global thresholds"/>
<node CREATED="1700000000000" ID="ID_3_1_1_2" MODIFIED="1700000000000" TEXT="Adapt based on historical consistency"/>
<node CREATED="1700000000000" ID="ID_3_1_1_3" MODIFIED="1700000000000" TEXT="Separate institutional variability from attacks"/>
</node>
</node>
<node CREATED="1700000000000" ID="ID_3_2" MODIFIED="1700000000000" TEXT="Threat Model">
<node CREATED="1700000000000" ID="ID_3_2_1" MODIFIED="1700000000000" TEXT="N healthcare institutions with private datasets"/>
<node CREATED="1700000000000" ID="ID_3_2_2" MODIFIED="1700000000000" TEXT="Fraction α &lt; 0.5 potentially Byzantine"/>
<node CREATED="1700000000000" ID="ID_3_2_3" MODIFIED="1700000000000" TEXT="Byzantine behaviors">
<node CREATED="1700000000000" ID="ID_3_2_3_1" MODIFIED="1700000000000" TEXT="Intentional: gradient manipulation, poisoning"/>
<node CREATED="1700000000000" ID="ID_3_2_3_2" MODIFIED="1700000000000" TEXT="Unintentional: hardware faults, data corruption"/>
</node>
</node>
<node CREATED="1700000000000" ID="ID_3_3" MODIFIED="1700000000000" TEXT="Temporal Behavioral Profile H_i(t)">
<node CREATED="1700000000000" ID="ID_3_3_1" MODIFIED="1700000000000" TEXT="1. Historical Gradient Norms">
<node CREATED="1700000000000" ID="ID_3_3_1_1" MODIFIED="1700000000000" TEXT="{||g_i(k)||_2} over window W"/>
</node>
<node CREATED="1700000000000" ID="ID_3_3_2" MODIFIED="1700000000000" TEXT="2. Directional Consistency">
<node CREATED="1700000000000" ID="ID_3_3_2_1" MODIFIED="1700000000000" TEXT="cos(g_i(k), g_i(j)) for recent pairs"/>
</node>
<node CREATED="1700000000000" ID="ID_3_3_3" MODIFIED="1700000000000" TEXT="3. EWMA Statistics">
<node CREATED="1700000000000" ID="ID_3_3_3_1" MODIFIED="1700000000000" TEXT="Mean: μ_i(t) = α·||g_i(t)||_2 + (1-α)·μ_i(t-1)"/>
<node CREATED="1700000000000" ID="ID_3_3_3_2" MODIFIED="1700000000000" TEXT="StdDev: σ_i(t) = sqrt formula"/>
<node CREATED="1700000000000" ID="ID_3_3_3_3" MODIFIED="1700000000000" TEXT="Decay parameter α controls emphasis"/>
</node>
<node CREATED="1700000000000" ID="ID_3_3_4" MODIFIED="1700000000000" TEXT="Captures unique gradient signature"/>
</node>
<node CREATED="1700000000000" ID="ID_3_4" MODIFIED="1700000000000" TEXT="Bootstrap Phase">
<node CREATED="1700000000000" ID="ID_3_4_1" MODIFIED="1700000000000" TEXT="Purpose: Prevent early profile contamination"/>
<node CREATED="1700000000000" ID="ID_3_4_2" MODIFIED="1700000000000" TEXT="Method: Conservative uniform clipping"/>
<node CREATED="1700000000000" ID="ID_3_4_3" MODIFIED="1700000000000" TEXT="Gradual transition to client-specific"/>
<node CREATED="1700000000000" ID="ID_3_4_4" MODIFIED="1700000000000" TEXT="Requires: empirical tuning of duration, stability, criteria"/>
</node>
<node CREATED="1700000000000" ID="ID_3_5" MODIFIED="1700000000000" TEXT="Multi-Dimensional Anomaly Detection">
<node CREATED="1700000000000" ID="ID_3_5_1" MODIFIED="1700000000000" TEXT="1. Magnitude Anomaly A_mag(i,t)">
<node CREATED="1700000000000" ID="ID_3_5_1_1" MODIFIED="1700000000000" TEXT="Formula: |||g_i(t)||_2 - μ_i(t-1)| / (σ_i(t-1) + ε)"/>
<node CREATED="1700000000000" ID="ID_3_5_1_2" MODIFIED="1700000000000" TEXT="Detects: Scaling attacks (ALIE)"/>
<node CREATED="1700000000000" ID="ID_3_5_1_3" MODIFIED="1700000000000" TEXT="Measures deviation from EWMA norms"/>
</node>
<node CREATED="1700000000000" ID="ID_3_5_2" MODIFIED="1700000000000" TEXT="2. Directional Anomaly A_dir(i,t)">
<node CREATED="1700000000000" ID="ID_3_5_2_1" MODIFIED="1700000000000" TEXT="Formula: 1 - (1/W)·Σcos(g_i(t), g_i(k))"/>
<node CREATED="1700000000000" ID="ID_3_5_2_2" MODIFIED="1700000000000" TEXT="Detects: Inner product manipulation"/>
<node CREATED="1700000000000" ID="ID_3_5_2_3" MODIFIED="1700000000000" TEXT="Captures sudden directional shifts"/>
</node>
<node CREATED="1700000000000" ID="ID_3_5_3" MODIFIED="1700000000000" TEXT="3. Temporal Anomaly A_temp(i,t)">
<node CREATED="1700000000000" ID="ID_3_5_3_1" MODIFIED="1700000000000" TEXT="Formula: |σ_i(t) - σ_i(t-W)| / (σ_i(t-W) + ε)"/>
<node CREATED="1700000000000" ID="ID_3_5_3_2" MODIFIED="1700000000000" TEXT="Detects: Distribution changes"/>
<node CREATED="1700000000000" ID="ID_3_5_3_3" MODIFIED="1700000000000" TEXT="Measures variance drift over time"/>
</node>
<node CREATED="1700000000000" ID="ID_3_5_4" MODIFIED="1700000000000" TEXT="4. Composite Score A_i(t)">
<node CREATED="1700000000000" ID="ID_3_5_4_1" MODIFIED="1700000000000" TEXT="Formula: sqrt(Σ λ_j·(A_j(i,t))^2)"/>
<node CREATED="1700000000000" ID="ID_3_5_4_2" MODIFIED="1700000000000" TEXT="j ∈ {mag, dir, temp}"/>
<node CREATED="1700000000000" ID="ID_3_5_4_3" MODIFIED="1700000000000" TEXT="Weights λ_j require experimental optimization"/>
<node CREATED="1700000000000" ID="ID_3_5_4_4" MODIFIED="1700000000000" TEXT="Forces attackers to appear normal in ALL dimensions"/>
</node>
</node>
<node CREATED="1700000000000" ID="ID_3_6" MODIFIED="1700000000000" TEXT="Client-Specific Thresholding">
<node CREATED="1700000000000" ID="ID_3_6_1" MODIFIED="1700000000000" TEXT="Threshold Function">
<node CREATED="1700000000000" ID="ID_3_6_1_1" MODIFIED="1700000000000" TEXT="τ_i(t) = μ_global(t) · f(A_i(t), R_i(t))"/>
<node CREATED="1700000000000" ID="ID_3_6_1_2" MODIFIED="1700000000000" TEXT="μ_global = median norm across clients"/>
<node CREATED="1700000000000" ID="ID_3_6_1_3" MODIFIED="1700000000000" TEXT="f decreases with high anomaly (stricter clipping)"/>
<node CREATED="1700000000000" ID="ID_3_6_1_4" MODIFIED="1700000000000" TEXT="f increases with high reliability (more flexibility)"/>
</node>
<node CREATED="1700000000000" ID="ID_3_6_2" MODIFIED="1700000000000" TEXT="Reliability Score R_i(t)">
<node CREATED="1700000000000" ID="ID_3_6_2_1" MODIFIED="1700000000000" TEXT="EWMA of benign behavior indicators"/>
<node CREATED="1700000000000" ID="ID_3_6_2_2" MODIFIED="1700000000000" TEXT="R_i(t) = γ·𝟙(A_i(t) &lt; τ_anomaly) + (1-γ)·R_i(t-1)"/>
<node CREATED="1700000000000" ID="ID_3_6_2_3" MODIFIED="1700000000000" TEXT="γ ∈ (0,1) smoothing parameter"/>
</node>
<node CREATED="1700000000000" ID="ID_3_6_3" MODIFIED="1700000000000" TEXT="Adapts based on behavioral consistency"/>
</node>
<node CREATED="1700000000000" ID="ID_3_7" MODIFIED="1700000000000" TEXT="Key Challenge">
<node CREATED="1700000000000" ID="ID_3_7_1" MODIFIED="1700000000000" TEXT="Integration of new institutions"/>
<node CREATED="1700000000000" ID="ID_3_7_2" MODIFIED="1700000000000" TEXT="Balance: security vs inclusivity"/>
<node CREATED="1700000000000" ID="ID_3_7_3" MODIFIED="1700000000000" TEXT="Graduated trust establishment process"/>
</node>
</node>
<node CREATED="1700000000000" ID="ID_4" MODIFIED="1700000000000" POSITION="right" TEXT="4. Research Hypotheses">
<node CREATED="1700000000000" ID="ID_4_1" MODIFIED="1700000000000" TEXT="H1: Heterogeneity Preservation">
<node CREATED="1700000000000" ID="ID_4_1_1" MODIFIED="1700000000000" TEXT="Client-specific profiles reduce false positives"/>
<node CREATED="1700000000000" ID="ID_4_1_2" MODIFIED="1700000000000" TEXT="Compared to global threshold methods"/>
<node CREATED="1700000000000" ID="ID_4_1_3" MODIFIED="1700000000000" TEXT="Under heterogeneous data distributions"/>
<node CREATED="1700000000000" ID="ID_4_1_4" MODIFIED="1700000000000" TEXT="Maintain/improve Byzantine detection rates"/>
</node>
<node CREATED="1700000000000" ID="ID_4_2" MODIFIED="1700000000000" TEXT="H2: Multi-Dimensional Defense">
<node CREATED="1700000000000" ID="ID_4_2_1" MODIFIED="1700000000000" TEXT="Combining magnitude + directional + temporal"/>
<node CREATED="1700000000000" ID="ID_4_2_2" MODIFIED="1700000000000" TEXT="More robust than single-metric approaches"/>
<node CREATED="1700000000000" ID="ID_4_2_3" MODIFIED="1700000000000" TEXT="Against adaptive adversaries"/>
<node CREATED="1700000000000" ID="ID_4_2_4" MODIFIED="1700000000000" TEXT="Forces normal behavior in ALL dimensions"/>
</node>
<node CREATED="1700000000000" ID="ID_4_3" MODIFIED="1700000000000" TEXT="H3: Temporal Discrimination">
<node CREATED="1700000000000" ID="ID_4_3_1" MODIFIED="1700000000000" TEXT="Window-based profiling distinguishes:"/>
<node CREATED="1700000000000" ID="ID_4_3_2" MODIFIED="1700000000000" TEXT="Abrupt Byzantine attacks"/>
<node CREATED="1700000000000" ID="ID_4_3_3" MODIFIED="1700000000000" TEXT="vs Gradual legitimate institutional changes"/>
<node CREATED="1700000000000" ID="ID_4_3_4" MODIFIED="1700000000000" TEXT="Addresses over/under-reaction problem"/>
</node>
<node CREATED="1700000000000" ID="ID_4_4" MODIFIED="1700000000000" TEXT="Core Tension">
<node CREATED="1700000000000" ID="ID_4_4_1" MODIFIED="1700000000000" TEXT="Maintain security"/>
<node CREATED="1700000000000" ID="ID_4_4_2" MODIFIED="1700000000000" TEXT="WITHOUT excluding legitimate diversity"/>
</node>
</node>
<node CREATED="1700000000000" ID="ID_5" MODIFIED="1700000000000" POSITION="left" TEXT="5. Experimental Design">
<node CREATED="1700000000000" ID="ID_5_1" MODIFIED="1700000000000" TEXT="Datasets">
<node CREATED="1700000000000" ID="ID_5_1_1" MODIFIED="1700000000000" TEXT="MIMIC-III">
<node CREATED="1700000000000" ID="ID_5_1_1_1" MODIFIED="1700000000000" TEXT="ICU mortality prediction"/>
<node CREATED="1700000000000" ID="ID_5_1_1_2" MODIFIED="1700000000000" TEXT="n = 49,785 patients"/>
</node>
<node CREATED="1700000000000" ID="ID_5_1_2" MODIFIED="1700000000000" TEXT="ChestX-ray8">
<node CREATED="1700000000000" ID="ID_5_1_2_1" MODIFIED="1700000000000" TEXT="Multi-label disease classification"/>
<node CREATED="1700000000000" ID="ID_5_1_2_2" MODIFIED="1700000000000" TEXT="108,948 images"/>
</node>
<node CREATED="1700000000000" ID="ID_5_1_3" MODIFIED="1700000000000" TEXT="ISIC 2019">
<node CREATED="1700000000000" ID="ID_5_1_3_1" MODIFIED="1700000000000" TEXT="Melanoma detection"/>
<node CREATED="1700000000000" ID="ID_5_1_3_2" MODIFIED="1700000000000" TEXT="n = 2,750 samples"/>
</node>
</node>
<node CREATED="1700000000000" ID="ID_5_2" MODIFIED="1700000000000" TEXT="Federated Setup">
<node CREATED="1700000000000" ID="ID_5_2_1" MODIFIED="1700000000000" TEXT="20 clients"/>
<node CREATED="1700000000000" ID="ID_5_2_2" MODIFIED="1700000000000" TEXT="Heterogeneity via Dirichlet allocation">
<node CREATED="1700000000000" ID="ID_5_2_2_1" MODIFIED="1700000000000" TEXT="α = 0.5 for label skew"/>
</node>
<node CREATED="1700000000000" ID="ID_5_2_3" MODIFIED="1700000000000" TEXT="Power law dataset sizes"/>
<node CREATED="1700000000000" ID="ID_5_2_4" MODIFIED="1700000000000" TEXT="Domain-specific augmentations"/>
</node>
<node CREATED="1700000000000" ID="ID_5_3" MODIFIED="1700000000000" TEXT="Attack Implementations">
<node CREATED="1700000000000" ID="ID_5_3_1" MODIFIED="1700000000000" TEXT="Untargeted Attacks">
<node CREATED="1700000000000" ID="ID_5_3_1_1" MODIFIED="1700000000000" TEXT="Random noise"/>
<node CREATED="1700000000000" ID="ID_5_3_1_2" MODIFIED="1700000000000" TEXT="Sign flipping"/>
</node>
<node CREATED="1700000000000" ID="ID_5_3_2" MODIFIED="1700000000000" TEXT="Targeted Attacks">
<node CREATED="1700000000000" ID="ID_5_3_2_1" MODIFIED="1700000000000" TEXT="ALIE (Baruch et al., 2019)"/>
<node CREATED="1700000000000" ID="ID_5_3_2_2" MODIFIED="1700000000000" TEXT="Inner Product Manipulation (Fang et al., 2020)"/>
</node>
<node CREATED="1700000000000" ID="ID_5_3_3" MODIFIED="1700000000000" TEXT="Adaptive Attacks">
<node CREATED="1700000000000" ID="ID_5_3_3_1" MODIFIED="1700000000000" TEXT="Slow-drift poisoning"/>
<node CREATED="1700000000000" ID="ID_5_3_3_2" MODIFIED="1700000000000" TEXT="Profile-aware strategies"/>
</node>
<node CREATED="1700000000000" ID="ID_5_3_4" MODIFIED="1700000000000" TEXT="Byzantine Fractions">
<node CREATED="1700000000000" ID="ID_5_3_4_1" MODIFIED="1700000000000" TEXT="20% to 40%"/>
<node CREATED="1700000000000" ID="ID_5_3_4_2" MODIFIED="1700000000000" TEXT="Independent and coordinated"/>
</node>
</node>
<node CREATED="1700000000000" ID="ID_5_4" MODIFIED="1700000000000" TEXT="Evaluation Metrics">
<node CREATED="1700000000000" ID="ID_5_4_1" MODIFIED="1700000000000" TEXT="Model accuracy"/>
<node CREATED="1700000000000" ID="ID_5_4_2" MODIFIED="1700000000000" TEXT="False positive rate (H1)"/>
<node CREATED="1700000000000" ID="ID_5_4_3" MODIFIED="1700000000000" TEXT="Attack impact (H2)"/>
<node CREATED="1700000000000" ID="ID_5_4_4" MODIFIED="1700000000000" TEXT="Detection latency (H3)"/>
</node>
<node CREATED="1700000000000" ID="ID_5_5" MODIFIED="1700000000000" TEXT="Baseline Comparisons">
<node CREATED="1700000000000" ID="ID_5_5_1" MODIFIED="1700000000000" TEXT="FedAvg (McMahan et al., 2017)"/>
<node CREATED="1700000000000" ID="ID_5_5_2" MODIFIED="1700000000000" TEXT="Krum (Blanchard et al., 2017)"/>
<node CREATED="1700000000000" ID="ID_5_5_3" MODIFIED="1700000000000" TEXT="Trimmed Mean (Yin et al., 2018)"/>
<node CREATED="1700000000000" ID="ID_5_5_4" MODIFIED="1700000000000" TEXT="ARC (Baruch et al., 2019)"/>
<node CREATED="1700000000000" ID="ID_5_5_5" MODIFIED="1700000000000" TEXT="FLTrust (Cao et al., 2021)"/>
<node CREATED="1700000000000" ID="ID_5_5_6" MODIFIED="1700000000000" TEXT="LASA (Xu et al., 2024)"/>
</node>
<node CREATED="1700000000000" ID="ID_5_6" MODIFIED="1700000000000" TEXT="Expected Benefits">
<node CREATED="1700000000000" ID="ID_5_6_1" MODIFIED="1700000000000" TEXT="High heterogeneity scenarios"/>
<node CREATED="1700000000000" ID="ID_5_6_2" MODIFIED="1700000000000" TEXT="20-30% Byzantine fractions"/>
</node>
</node>
<node CREATED="1700000000000" ID="ID_6" MODIFIED="1700000000000" POSITION="left" TEXT="6. Key Contributions">
<node CREATED="1700000000000" ID="ID_6_1" MODIFIED="1700000000000" TEXT="1. Client-Adaptive Framework">
<node CREATED="1700000000000" ID="ID_6_1_1" MODIFIED="1700000000000" TEXT="Node-specific behavioral profiles"/>
<node CREATED="1700000000000" ID="ID_6_1_2" MODIFIED="1700000000000" TEXT="First practical implementation of Werner et al. (2023) theory"/>
</node>
<node CREATED="1700000000000" ID="ID_6_2" MODIFIED="1700000000000" TEXT="2. Multi-Dimensional Detection">
<node CREATED="1700000000000" ID="ID_6_2_1" MODIFIED="1700000000000" TEXT="Magnitude + Direction + Temporal"/>
<node CREATED="1700000000000" ID="ID_6_2_2" MODIFIED="1700000000000" TEXT="Comprehensive anomaly coverage"/>
</node>
<node CREATED="1700000000000" ID="ID_6_3" MODIFIED="1700000000000" TEXT="3. Adaptive Thresholding">
<node CREATED="1700000000000" ID="ID_6_3_1" MODIFIED="1700000000000" TEXT="Balances responsiveness with stability"/>
</node>
<node CREATED="1700000000000" ID="ID_6_4" MODIFIED="1700000000000" TEXT="4. Healthcare-Specific Validation">
<node CREATED="1700000000000" ID="ID_6_4_1" MODIFIED="1700000000000" TEXT="Comprehensive experimental design"/>
<node CREATED="1700000000000" ID="ID_6_4_2" MODIFIED="1700000000000" TEXT="Three diverse medical datasets"/>
</node>
</node>
<node CREATED="1700000000000" ID="ID_7" MODIFIED="1700000000000" POSITION="left" TEXT="7. Limitations &amp; Future Work">
<node CREATED="1700000000000" ID="ID_7_1" MODIFIED="1700000000000" TEXT="Current Limitations">
<node CREATED="1700000000000" ID="ID_7_1_1" MODIFIED="1700000000000" TEXT="Vulnerability to slow-drift attacks"/>
<node CREATED="1700000000000" ID="ID_7_1_2" MODIFIED="1700000000000" TEXT="Potential manipulation by colluding clients"/>
<node CREATED="1700000000000" ID="ID_7_1_3" MODIFIED="1700000000000" TEXT="Scalability for thousands of participants"/>
<node CREATED="1700000000000" ID="ID_7_1_4" MODIFIED="1700000000000" TEXT="Privacy risks from behavioral profiles"/>
</node>
<node CREATED="1700000000000" ID="ID_7_2" MODIFIED="1700000000000" TEXT="Future Directions">
<node CREATED="1700000000000" ID="ID_7_2_1" MODIFIED="1700000000000" TEXT="Formal convergence guarantees"/>
<node CREATED="1700000000000" ID="ID_7_2_2" MODIFIED="1700000000000" TEXT="Differential privacy integration"/>
<node CREATED="1700000000000" ID="ID_7_2_3" MODIFIED="1700000000000" TEXT="Hierarchical aggregation for scale"/>
<node CREATED="1700000000000" ID="ID_7_2_4" MODIFIED="1700000000000" TEXT="Cross-domain validation beyond healthcare"/>
</node>
</node>
<node CREATED="1700000000000" ID="ID_8" MODIFIED="1700000000000" POSITION="left" TEXT="8. Comparison Table">
<node CREATED="1700000000000" ID="ID_8_1" MODIFIED="1700000000000" TEXT="Defense Capabilities Matrix">
<node CREATED="1700000000000" ID="ID_8_1_1" MODIFIED="1700000000000" TEXT="Magnitude Defense">
<node CREATED="1700000000000" ID="ID_8_1_1_1" MODIFIED="1700000000000" TEXT="Most methods: ✓"/>
<node CREATED="1700000000000" ID="ID_8_1_1_2" MODIFIED="1700000000000" TEXT="CAAC-FL: ✓ (norm + EWMA)"/>
</node>
<node CREATED="1700000000000" ID="ID_8_1_2" MODIFIED="1700000000000" TEXT="Directional Defense">
<node CREATED="1700000000000" ID="ID_8_1_2_1" MODIFIED="1700000000000" TEXT="Median/Krum/ARC: ✗"/>
<node CREATED="1700000000000" ID="ID_8_1_2_2" MODIFIED="1700000000000" TEXT="FLTrust/Clipped/LASA: ✓"/>
<node CREATED="1700000000000" ID="ID_8_1_2_3" MODIFIED="1700000000000" TEXT="CAAC-FL: ✓ (cosine + median)"/>
</node>
<node CREATED="1700000000000" ID="ID_8_1_3" MODIFIED="1700000000000" TEXT="Client-Specific">
<node CREATED="1700000000000" ID="ID_8_1_3_1" MODIFIED="1700000000000" TEXT="Most methods: ✗"/>
<node CREATED="1700000000000" ID="ID_8_1_3_2" MODIFIED="1700000000000" TEXT="LASA: ✓ (sparsification)"/>
<node CREATED="1700000000000" ID="ID_8_1_3_3" MODIFIED="1700000000000" TEXT="BR-MTRL: ✓ (final layer)"/>
<node CREATED="1700000000000" ID="ID_8_1_3_4" MODIFIED="1700000000000" TEXT="CAAC-FL: ✓ (node-level)"/>
</node>
<node CREATED="1700000000000" ID="ID_8_1_4" MODIFIED="1700000000000" TEXT="Temporal">
<node CREATED="1700000000000" ID="ID_8_1_4_1" MODIFIED="1700000000000" TEXT="All others: ✗"/>
<node CREATED="1700000000000" ID="ID_8_1_4_2" MODIFIED="1700000000000" TEXT="CAAC-FL: ✓ (EWMA)"/>
</node>
<node CREATED="1700000000000" ID="ID_8_1_5" MODIFIED="1700000000000" TEXT="Adaptive">
<node CREATED="1700000000000" ID="ID_8_1_5_1" MODIFIED="1700000000000" TEXT="Clipped Clustering: ✓ (global)"/>
<node CREATED="1700000000000" ID="ID_8_1_5_2" MODIFIED="1700000000000" TEXT="LASA: ✓ (layer)"/>
<node CREATED="1700000000000" ID="ID_8_1_5_3" MODIFIED="1700000000000" TEXT="CAAC-FL: ✓ (exponential)"/>
</node>
</node>
</node>
<node CREATED="1700000000000" ID="ID_9" MODIFIED="1700000000000" POSITION="left" TEXT="9. Key Insights">
<node CREATED="1700000000000" ID="ID_9_1" MODIFIED="1700000000000" TEXT="Paradigm Shift">
<node CREATED="1700000000000" ID="ID_9_1_1" MODIFIED="1700000000000" TEXT="From global thresholds"/>
<node CREATED="1700000000000" ID="ID_9_1_2" MODIFIED="1700000000000" TEXT="To context-aware defense"/>
</node>
<node CREATED="1700000000000" ID="ID_9_2" MODIFIED="1700000000000" TEXT="Security + Inclusivity">
<node CREATED="1700000000000" ID="ID_9_2_1" MODIFIED="1700000000000" TEXT="Not mutually exclusive"/>
<node CREATED="1700000000000" ID="ID_9_2_2" MODIFIED="1700000000000" TEXT="Both paramount in healthcare"/>
</node>
<node CREATED="1700000000000" ID="ID_9_3" MODIFIED="1700000000000" TEXT="Broader Implications">
<node CREATED="1700000000000" ID="ID_9_3_1" MODIFIED="1700000000000" TEXT="Any heterogeneous FL deployment"/>
<node CREATED="1700000000000" ID="ID_9_3_2" MODIFIED="1700000000000" TEXT="Where diversity is legitimate feature"/>
<node CREATED="1700000000000" ID="ID_9_3_3" MODIFIED="1700000000000" TEXT="Not a bug to be filtered out"/>
</node>
</node>
</node>
</map>
