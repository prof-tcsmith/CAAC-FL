# CAAC-FL: WITS 2025 Presentation Materials

## Paper: Distinguishing Medical Diversity from Byzantine Attacks: Client-Adaptive Anomaly Detection for Healthcare Federated Learning

**Authors:** Timothy Smith, Anol Bhattacherjee, and Raghu Ram Komara (University of South Florida)

---

# PART I: PRESENTATION SLIDES (10-15 Minutes)

---

## Slide 1: Title Slide

**Distinguishing Medical Diversity from Byzantine Attacks:**
**Client-Adaptive Anomaly Detection for Healthcare Federated Learning**

Timothy Smith, Anol Bhattacherjee, and Raghu Ram Komara
University of South Florida

Workshop on Information Technologies and Systems (WITS) 2025
*Research in Progress*

---

## Slide 2: The Promise of Federated Learning in Healthcare

**Federated Learning: Collaborative AI Without Sharing Data**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Hospital A │    │  Hospital B │    │  Hospital C │
│  (Pediatric)│    │  (Geriatric)│    │  (Oncology) │
│     Data    │    │     Data    │    │     Data    │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       │   Local Model    │   Local Model    │
       │    Updates ↓     │    Updates ↓     │
       └──────────────────┼──────────────────┘
                          ↓
              ┌───────────────────┐
              │   Central Server  │
              │  Aggregates into  │
              │   Global Model    │
              └───────────────────┘
```

**Key Benefits:**
- Privacy preservation (HIPAA, GDPR compliance)
- Leverage diverse institutional data
- No centralized data storage required

*McMahan et al., 2017 (89,000+ citations)*

---

## Slide 3: Brief History of Federated Learning

**Evolution Timeline:**

| Year | Milestone | Contribution |
|------|-----------|--------------|
| 1982 | Byzantine Generals Problem | Foundational distributed consensus (Lamport et al.) |
| 2016-17 | FedAvg Introduced | First practical FL algorithm (McMahan et al., Google) |
| 2017 | Krum | First Byzantine-tolerant gradient descent (Blanchard et al.) |
| 2018 | Trimmed Mean | Optimal statistical rates for robust aggregation (Yin et al.) |
| 2019 | ALIE Attack | Showed small perturbations evade defenses (Baruch et al.) |
| 2020 | FLTrust | Trust bootstrapping with server data (Cao et al.) |
| 2021-23 | Adaptive Methods | RFA, ARC, Layer-wise approaches |
| 2024 | Reality Check | Li et al. show defenses fail on non-IID data |
| 2025 | **Our Work** | Client-adaptive temporal profiling |

---

## Slide 4: The Byzantine Threat

**What is a Byzantine Attack?**

Named after the "Byzantine Generals Problem" (Lamport et al., 1982)

**In Federated Learning Context:**
- Compromised or faulty clients submit **corrupted gradient updates**
- Can be **intentional** (malicious actors) or **unintentional** (hardware failures)
- Even **one Byzantine participant** can catastrophically degrade model performance

**Healthcare Stakes:**
- Model errors directly impact patient outcomes
- A single compromised hospital could increase misdiagnosis rates
- Trust is essential but verification is difficult

---

## Slide 5: Known Byzantine Attack Strategies

**Attack Taxonomy:**

| Attack Type | Description | Reference |
|-------------|-------------|-----------|
| **Sign Flipping** | Reverse gradient direction | Basic attack |
| **Random Noise** | Add Gaussian noise to updates | Basic attack |
| **ALIE** | "A Little Is Enough" - small changes within variance | Baruch et al., 2019 |
| **IPM** | Inner Product Manipulation - make aggregated gradient negative | Xie et al., 2020 |
| **Label Flipping** | Poison training labels | Data poisoning |
| **Backdoor** | Insert hidden triggers | Bagdasaryan et al., 2020 |
| **Slow Drift** | Gradual poisoning over time | Adaptive attack |

**Key Insight:** Sophisticated attacks stay within "normal" statistical bounds

---

## Slide 6: Current Defense Approaches

**Three Main Categories:**

**1. Statistical Filtering**
- *Trimmed Mean* (Yin et al., 2018): Remove outliers coordinate-wise
- *Median*: Use median instead of mean
- **Limitation:** Fails when attacks appear normal in most dimensions

**2. Geometric Methods**
- *Krum* (Blanchard et al., 2017): Select update closest to neighbors
- *Multi-Krum*: Select multiple updates
- **Limitation:** Filters legitimate diverse contributions

**3. Trust-Based Methods**
- *FLTrust* (Cao et al., 2021): Server maintains "root dataset"
- **Limitation:** Violates FL's core privacy principle

---

## Slide 7: The Critical Problem - Heterogeneity vs. Attacks

**The Fundamental Tension:**

```
    Legitimate Diversity          vs.        Byzantine Attack
    ──────────────────                       ─────────────────

    Pediatric Hospital                       Compromised Client
         ↓                                        ↓
    Different gradient                      Different gradient
    (normal for them)                       (malicious intent)
         ↓                                        ↓
    LOOKS IDENTICAL TO EXISTING DEFENSES
```

**Real Evidence (Li et al., 2024):**
> "With Non-IID data, some aggregation schemes achieve **less than 10% accuracy** even in the **complete absence of Byzantine clients**"

*Existing defenses cannot distinguish legitimate diversity from attacks!*

---

## Slide 8: Why Existing Methods Fail

**The Global Threshold Problem:**

```
                    Global Threshold Applied Uniformly
                              ↓
    ┌─────────────────────────────────────────────────┐
    │                 ACCEPTED                        │
    │    ○ Normal      ○ Normal     ○ Normal         │
    │    Hospital A    Hospital B   Hospital C        │
    ├─────────────────────────────────────────────────┤
    │                 REJECTED                        │
    │    ○ Pediatric   ✕ Byzantine                   │
    │    Hospital D    Attacker                       │
    └─────────────────────────────────────────────────┘

    Problem: Specialized hospitals filtered out with attackers!
```

**Consequences:**
- Valuable specialized data excluded
- Model loses diversity benefits
- Healthcare outcomes suffer

---

## Slide 9: Our Solution - CAAC-FL Overview

**Client-Adaptive Anomaly-Aware Clipping for Federated Learning**

**Core Innovation: Client-Specific Behavioral Profiling**

```
Instead of:  Global Threshold → All Clients

CAAC-FL:    Client A Profile → Client A Threshold
            Client B Profile → Client B Threshold
            Client C Profile → Client C Threshold
            ...
```

**Key Principle:**
> "What is anomalous for *this specific client* based on *their historical behavior*?"

Not: "What is anomalous compared to the population mean?"

---

## Slide 10: CAAC-FL Technical Architecture

**Three-Dimensional Anomaly Detection:**

```
                    ┌───────────────────────┐
                    │   Composite Anomaly   │
                    │        Score          │
                    └───────────┬───────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ↓                     ↓                     ↓
    ┌───────────┐        ┌───────────┐        ┌───────────┐
    │ Magnitude │        │Directional│        │ Temporal  │
    │  Anomaly  │        │  Anomaly  │        │Consistency│
    │           │        │           │        │           │
    │ ||g|| vs  │        │ cos(g_t,  │        │  Variance │
    │  history  │        │   g_t-1)  │        │   drift   │
    └───────────┘        └───────────┘        └───────────┘
         ↑                     ↑                     ↑
         └─────────────────────┼─────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │ Client Historical   │
                    │ Behavioral Profile  │
                    │  (EWMA Statistics)  │
                    └─────────────────────┘
```

---

## Slide 11: Why Three Dimensions?

**Multi-Dimensional Defense Makes Evasion Harder:**

| Dimension | Catches | Attack Example |
|-----------|---------|----------------|
| **Magnitude** | Scaling attacks | ALIE (small changes) |
| **Directional** | Steering attempts | IPM (direction manipulation) |
| **Temporal** | Sudden behavioral changes | Slow drift poisoning |

**Attacker Must:**
- Appear normal across **all three metrics simultaneously**
- Match client's **individual historical pattern**
- Maintain consistency over **time window**

*Much harder than fooling a single global threshold!*

---

## Slide 12: Temporal Behavioral Profiling

**Exponentially Weighted Moving Average (EWMA):**

```
For each client i at round t:

Mean:     μ_t = α · ||g_t|| + (1-α) · μ_{t-1}

Variance: σ_t = α · (||g_t|| - μ_t)² + (1-α) · σ²_{t-1}
```

**Why EWMA?**
- Captures each client's unique "gradient signature"
- Reflects their data distribution and clinical focus
- Distinguishes consistent patterns from anomalous deviations
- Decay parameter α balances recent vs. historical behavior

**Result:** Client-specific baselines that evolve appropriately

---

## Slide 13: Adaptive Clipping Thresholds

**Threshold Function:**

```
τ_i^t = μ_global^t · f(A_i^t, R_i^t)
```

Where:
- μ_global^t = Global median norm (anchor point)
- A_i^t = Client's anomaly score
- R_i^t = Client's historical reliability score

**Behavior:**
- **High anomaly score** → Stricter clipping (suspicious)
- **High reliability score** → More flexibility (trusted)
- **New clients** → Conservative defaults during bootstrap

---

## Slide 14: How CAAC-FL Handles Heterogeneity

**Scenario: Pediatric vs. Geriatric Hospitals**

| Hospital | Patient Population | Expected Gradients | CAAC-FL Response |
|----------|-------------------|-------------------|------------------|
| Pediatric | Children | Different from population | *Learn their normal pattern* |
| Geriatric | Elderly | Different from population | *Learn their normal pattern* |
| Compromised | N/A | Deviates from *their own* history | *Flag as anomalous* |

**Key Difference from Existing Methods:**
- Krum/Trimmed Mean: Reject pediatric hospital for being "too different"
- CAAC-FL: Accept pediatric hospital (consistent with their history)

---

## Slide 15: Research Hypotheses

**H1: Heterogeneity Preservation**
> Client-specific behavioral profiles will significantly reduce false positive rates compared to global threshold methods while maintaining comparable or better Byzantine detection rates.

**H2: Multi-Dimensional Defense**
> Combining magnitude, directional, and temporal anomaly metrics will provide more robust Byzantine detection than single-metric approaches.

**H3: Temporal Discrimination**
> The window-based profiling approach will successfully distinguish between abrupt Byzantine attacks and gradual legitimate institutional changes.

---

## Slide 16: Experimental Design

**Datasets:**
| Dataset | Task | Size |
|---------|------|------|
| MIMIC-III | ICU mortality prediction | 49,785 |
| ChestX-ray8 | Multi-label disease classification | 108,948 |
| ISIC 2019 | Melanoma detection | 2,750 |

**Heterogeneity Simulation:**
- Dirichlet allocation (α=0.5) for label skew
- Power law dataset sizes
- Domain-specific augmentations

**Attack Scenarios:**
- 20-40% Byzantine fraction
- Untargeted (noise, sign flip) and targeted (ALIE, IPM)
- Adaptive attacks (slow-drift, profile-aware)

---

## Slide 17: Comparison Methods

**Baselines:**
| Method | Year | Category |
|--------|------|----------|
| FedAvg | 2017 | No defense (baseline) |
| Krum | 2017 | Geometric |
| Trimmed Mean | 2018 | Statistical |
| ARC | 2019 | Adaptive clipping |
| FLTrust | 2021 | Trust-based |
| LASA | 2024 | Layer-adaptive |

**Metrics:**
- Model accuracy
- False positive rate (H1)
- Attack impact reduction (H2)
- Detection latency (H3)

---

## Slide 18: Novelty and Contributions

**What Makes CAAC-FL Different:**

| Feature | Prior Work | CAAC-FL |
|---------|------------|---------|
| Threshold Type | Global | Client-specific |
| Temporal Tracking | None | EWMA-based profiles |
| Anomaly Detection | Single dimension | Three dimensions |
| Heterogeneity Handling | Problematic | Core design goal |
| Theoretical Basis | Various | Werner et al., 2023 |

**First Practical Implementation of:**
- Client-specific behavioral profiling for Byzantine defense
- Multi-dimensional anomaly detection with temporal consistency
- Adaptive thresholds based on historical reliability

---

## Slide 19: Positioning in Literature

**Building on Theoretical Foundations:**

Werner et al. (2023) proved that client-specific adaptation enhances Byzantine robustness, but noted:
> "Practical implementations of this theory remain nascent"

**CAAC-FL bridges this gap** with:
- Concrete algorithmic instantiation
- Healthcare-specific validation
- Comprehensive experimental design

**Addressing Empirical Findings:**

Li et al. (2024) showed existing defenses fail on non-IID data.
**CAAC-FL directly addresses this** by treating heterogeneity as a feature, not a bug.

---

## Slide 20: Challenges and Limitations

**Known Challenges:**

1. **Slow-drift attacks**: Gradual poisoning may evade temporal detection
2. **Colluding Byzantine clients**: Could manipulate collective profiles
3. **Scalability**: Maintaining profiles for thousands of clients
4. **Privacy**: Detailed behavioral profiles could leak information
5. **Parameter sensitivity**: Bootstrap duration, decay rates need tuning

**Future Work:**
- Formal convergence guarantees
- Differential privacy integration
- Hierarchical aggregation for scale
- Cross-domain validation beyond healthcare

---

## Slide 21: Expected Impact

**For Healthcare:**
- Enable secure multi-hospital collaborations
- Preserve specialized institutional knowledge
- Improve model generalization across patient populations

**For Federated Learning Research:**
- Paradigm shift toward context-aware Byzantine defense
- Framework for balancing security and inclusivity
- Foundation for heterogeneous FL deployments

**For Practice:**
- Actionable approach for real-world FL systems
- Configurable parameters for different trust environments
- Applicable beyond healthcare to any heterogeneous domain

---

## Slide 22: Conclusion

**Summary:**

1. **Problem**: Byzantine defenses fail in heterogeneous settings (Li et al., 2024 showed <10% accuracy even without attacks)

2. **Insight**: The fundamental flaw is treating all clients as statistically identical

3. **Solution**: CAAC-FL introduces client-specific behavioral profiling with:
   - Temporal tracking via EWMA
   - Multi-dimensional anomaly detection
   - Adaptive clipping thresholds

4. **Status**: Research in progress with comprehensive experimental design

**Call to Action:**
- Feedback on experimental design welcome
- Interested in collaboration opportunities
- Contact: [authors' emails]

---

## Slide 23: Questions?

**Thank you!**

**Key Takeaway:**
> "What's anomalous for a pediatric hospital is different from what's anomalous for a geriatric center. CAAC-FL learns and respects these differences."

**Contact Information:**
Timothy Smith, Anol Bhattacherjee, Raghu Ram Komara
University of South Florida

---

# PART II: DETAILED SPEAKER NOTES

---

## Slide 1 Notes (Title Slide) - 30 seconds

"Good [morning/afternoon]. I'm [Name] from the University of South Florida, and today I'll be presenting our research in progress on a critical challenge in healthcare federated learning: how do we protect against malicious actors without accidentally excluding the diverse, specialized medical institutions whose data we most need?"

---

## Slide 2 Notes (Promise of FL) - 1 minute

"Let me start by reminding everyone what makes federated learning so exciting for healthcare.

Traditional machine learning requires centralizing data - which is essentially impossible in healthcare due to HIPAA, GDPR, and institutional policies. Federated learning, introduced by McMahan and colleagues at Google in 2017, offers an elegant solution: instead of bringing data to the model, we bring the model to the data.

Each hospital trains on their local data and only shares model updates - gradients or weights - never the raw patient information. The central server aggregates these updates into a global model that benefits from the collective knowledge of all participating institutions.

This paper, by the way, has over 89,000 citations - it really sparked a revolution in privacy-preserving machine learning. But as we'll see, this distributed architecture introduces new security challenges."

---

## Slide 3 Notes (History) - 1 minute

"Let me briefly situate federated learning in its historical context.

The security challenge we're addressing - Byzantine failures - actually dates back to 1982, when Lamport introduced the Byzantine Generals Problem for distributed consensus. The name comes from an analogy about generals who can't trust all the messengers between them.

Federated learning emerged in 2016-2017 with FedAvg. Almost immediately, researchers recognized the Byzantine threat, and Blanchard introduced Krum in 2017 as the first provably Byzantine-tolerant approach.

The field has evolved significantly since then, but as we'll see, a major breakthrough in 2024 showed that all these defenses have a fundamental flaw when data isn't identically distributed - which is always the case in healthcare. Our work, CAAC-FL, addresses this head-on."

---

## Slide 4 Notes (Byzantine Threat) - 1 minute

"So what exactly is a Byzantine attack in federated learning?

The term comes from the Byzantine Generals Problem - imagine generals who need to agree on a battle plan, but some generals might be traitors sending conflicting messages. In our context, some clients might send corrupted gradient updates.

This can be intentional - a malicious actor compromising a hospital's system - or unintentional, like hardware failures or corrupted data. From the server's perspective, these are indistinguishable.

Here's what makes this critical for healthcare: even a single Byzantine participant in a multi-hospital network could cause significant increases in misdiagnosis rates. When your model is helping diagnose cancer or predict ICU mortality, this isn't an abstract security concern - it directly impacts patient outcomes."

---

## Slide 5 Notes (Attack Strategies) - 1 minute

"Researchers have documented a variety of Byzantine attack strategies, ranging from simple to sophisticated.

Basic attacks like sign flipping or random noise are easy to implement but also relatively easy to detect. The more concerning attacks are sophisticated ones like ALIE - 'A Little Is Enough' - which Baruch and colleagues showed in 2019. They demonstrated that attackers don't need to make large changes; small perturbations within the natural variance of honest gradients can evade detection while still corrupting the model.

Inner Product Manipulation, or IPM, is another clever attack that manipulates the direction of the aggregated gradient without appearing statistically unusual.

The key insight here is that sophisticated attackers specifically design their corrupted updates to stay within 'normal' statistical bounds. This is the arms race we're dealing with."

---

## Slide 6 Notes (Current Defenses) - 1.5 minutes

"Current defenses fall into three main categories, each with significant limitations.

Statistical filtering methods like Trimmed Mean, introduced by Yin and colleagues in 2018, remove outlier values before averaging. The problem? Sophisticated attacks like ALIE are designed to appear normal in most dimensions while being malicious in a few critical ones.

Geometric methods like Krum select the update that's closest to its neighbors. The intuition is that Byzantine updates will be isolated. But in healthcare, a specialized pediatric hospital's updates might legitimately be far from the geriatric center's - Krum can't tell the difference.

Trust-based methods like FLTrust, introduced by Cao and colleagues in 2021, are quite effective but require the server to maintain a 'root dataset' for comparison. This fundamentally violates federated learning's core privacy principle - we're doing FL precisely because we can't centralize data!

None of these approaches can handle the heterogeneity inherent in healthcare data."

---

## Slide 7 Notes (Critical Problem) - 1 minute

"This brings us to the critical problem that motivated our research.

In healthcare, legitimate diversity is the norm, not the exception. A pediatric hospital serves a fundamentally different patient population than a geriatric center. Their gradient updates will naturally look different - different diseases, different treatment patterns, different outcomes.

But to existing defenses, this legitimate diversity looks exactly like a Byzantine attack. Both produce gradients that deviate from the population mean.

Li and colleagues conducted a systematic experimental study in 2024 that quantified this problem. Their finding was shocking: with non-IID data - which is always the case in real federated learning - some aggregation schemes achieved less than 10% accuracy even in the complete absence of any Byzantine attackers.

Let me repeat that: the defenses themselves broke the model, without any attack occurring. This is unacceptable for healthcare applications."

---

## Slide 8 Notes (Why Methods Fail) - 45 seconds

"This diagram illustrates the fundamental problem. When you apply a global threshold uniformly across all clients, you're making an implicit assumption that all clients should look statistically similar.

Anything outside the threshold gets rejected - which correctly rejects Byzantine attackers, but also incorrectly rejects specialized hospitals that serve unique patient populations.

The pediatric hospital in this example gets filtered out along with the Byzantine attacker, simply because their updates are 'too different' from the majority. The model loses the benefit of their specialized knowledge, and potentially makes worse predictions for pediatric patients as a result."

---

## Slide 9 Notes (CAAC-FL Overview) - 1 minute

"Our solution, CAAC-FL - Client-Adaptive Anomaly-Aware Clipping for Federated Learning - takes a fundamentally different approach.

Instead of asking 'Is this update anomalous compared to the global population?' we ask 'Is this update anomalous for this specific client, given their historical behavior?'

Each client gets their own behavioral profile and their own threshold. A pediatric hospital is compared to their own history, not to the geriatric center. If they've always produced gradients with certain characteristics, that's their normal - and we expect it to continue.

An attacker, on the other hand, would need to compromise a client and then produce updates that match that client's specific historical pattern - a much harder task than simply staying within global statistical bounds."

---

## Slide 10 Notes (Technical Architecture) - 1.5 minutes

"Let me walk you through the technical architecture of CAAC-FL.

At the heart is three-dimensional anomaly detection. Rather than looking at just one metric, we examine updates from three perspectives:

First, magnitude anomaly - is the size of this gradient consistent with what this client usually sends? This catches scaling attacks.

Second, directional anomaly - is the direction of this gradient consistent with this client's typical updates? This catches steering attacks like IPM.

Third, temporal consistency - has this client's variance pattern changed significantly? This catches sudden behavioral shifts that might indicate a compromise.

All three dimensions draw from a client-specific historical behavioral profile, which we maintain using Exponentially Weighted Moving Averages. The EWMA approach gives us smooth, adaptive baselines that can evolve as institutions legitimately change over time, while still being sensitive to sudden anomalous shifts.

The key insight is that an attacker would need to appear normal across all three dimensions simultaneously, while also matching the specific historical pattern of the client they've compromised. That's a much higher bar than fooling a single global threshold."

---

## Slide 11 Notes (Why Three Dimensions) - 45 seconds

"Why do we need all three dimensions? Because different attack strategies target different aspects of the gradient.

ALIE attacks manipulate magnitude in subtle ways - magnitude detection catches these. IPM attacks manipulate direction - directional detection catches these. Slow drift attacks gradually shift behavior over time - temporal consistency detection catches these.

By requiring attackers to appear normal across all three simultaneously, and to match a specific client's historical pattern, we've significantly raised the difficulty of successful attacks.

It's like a three-factor authentication for Byzantine defense."

---

## Slide 12-13 Notes (Technical Details) - 1 minute

"The EWMA formulation captures how we maintain these behavioral profiles over time. The decay parameter alpha controls how much weight we give to recent observations versus historical behavior.

For the threshold function, we anchor to the global median norm - this provides stability - but then adjust based on each client's anomaly score and reliability history. Clients with consistently low anomaly scores build up reliability and get more flexibility. Clients with suspicious behavior get tighter scrutiny.

New clients start with conservative default thresholds during a bootstrap phase, and gradually transition to client-specific profiling as we accumulate history. This handles the challenge of onboarding new institutions securely."

---

## Slide 14 Notes (Handling Heterogeneity) - 45 seconds

"This table illustrates how CAAC-FL handles the heterogeneity scenario that breaks existing methods.

The pediatric hospital produces gradients that look different from the population - but they're consistent with that hospital's history. CAAC-FL recognizes this as normal for them.

Similarly for the geriatric hospital. Each institution has its own learned baseline.

When an attacker compromises a client, their updates deviate from that client's established pattern - that's what triggers the anomaly detection, not deviation from a global mean."

---

## Slide 15-17 Notes (Hypotheses and Experimental Design) - 1 minute

"Our three hypotheses directly address the gaps in the literature.

H1 tests whether client-specific profiles actually reduce false positives without sacrificing security. H2 tests whether combining multiple anomaly dimensions provides better defense than single metrics. H3 tests whether temporal profiling can distinguish sudden attacks from gradual legitimate evolution.

We're validating on three healthcare datasets - MIMIC-III for ICU mortality prediction, ChestX-ray8 for disease classification, and ISIC for melanoma detection. We simulate realistic heterogeneity using Dirichlet allocation and test against both simple and sophisticated attack strategies.

We'll compare against the major existing approaches: FedAvg as a baseline, Krum, Trimmed Mean, ARC, FLTrust, and the recent LASA approach."

---

## Slide 18-19 Notes (Novelty and Positioning) - 1 minute

"What makes CAAC-FL novel is the combination of client-specific adaptation with temporal behavioral profiling. Existing methods are either global threshold-based or, in the case of recent theoretical work like Werner et al. 2023, prove that client-specific adaptation helps but don't provide practical implementations.

Werner's paper specifically notes that 'practical implementations of this theory remain nascent.' CAAC-FL is our attempt to bridge that gap with a concrete, implementable algorithm.

We're also directly addressing the empirical findings of Li et al. 2024 that showed defenses failing on heterogeneous data. Rather than treating heterogeneity as a problem to overcome, CAAC-FL treats it as an expected feature of realistic federated learning."

---

## Slide 20 Notes (Challenges) - 1 minute

"We want to be transparent about the challenges and limitations of our approach.

Slow-drift attacks - where an attacker gradually shifts a client's behavior over many rounds - could potentially evade our temporal detection. The EWMA would slowly adapt to the corrupted behavior.

Colluding Byzantine clients could potentially coordinate to manipulate what appears 'normal' for a cluster of institutions.

Scalability is a practical concern - maintaining detailed behavioral profiles for thousands of clients adds overhead.

And ironically, the detailed behavioral profiles that enable our defense could themselves be a privacy risk if exposed.

These limitations define our future research agenda: formal convergence guarantees, differential privacy integration, hierarchical approaches for scale, and validation across domains."

---

## Slide 21-22 Notes (Impact and Conclusion) - 1 minute

"If successful, CAAC-FL would enable secure multi-hospital collaborations that preserve the specialized knowledge of each institution. This could significantly improve healthcare AI systems that currently struggle with heterogeneous patient populations.

More broadly, we see this as a paradigm shift in how we think about Byzantine defense - from 'identify the outliers' to 'understand each participant's context.'

To summarize: the problem is that existing defenses fail on heterogeneous data. Our insight is that the fundamental flaw is treating all clients identically. Our solution - CAAC-FL - introduces client-specific behavioral profiling with temporal tracking and multi-dimensional anomaly detection.

This is research in progress, and we welcome feedback on our experimental design and theoretical framing."

---

## Slide 23 Notes (Questions) - 30 seconds

"I'll leave you with this key takeaway: What's anomalous for a pediatric hospital is different from what's anomalous for a geriatric center. CAAC-FL learns and respects these differences, enabling secure collaboration without sacrificing diversity.

Thank you for your attention. I'm happy to take questions."

---

# PART III: ACTION PLAN TO COMPLETE THE RESEARCH

---

## Phase 1: Foundation and Infrastructure (Weeks 1-3)

### Task 1.1: Implementation Framework
- [ ] Set up modular FL simulation environment
- [ ] Implement base FedAvg with configurable parameters
- [ ] Create abstract Byzantine attack interface
- [ ] Implement baseline defense methods (Krum, Trimmed Mean, FLTrust)
- [ ] Establish reproducibility infrastructure (seed management, logging)

### Task 1.2: CAAC-FL Core Implementation
- [ ] Implement client behavioral profile data structure
- [ ] Implement EWMA statistics tracking
- [ ] Implement magnitude anomaly scoring
- [ ] Implement directional anomaly scoring
- [ ] Implement temporal consistency scoring
- [ ] Implement composite anomaly score computation
- [ ] Implement adaptive threshold function
- [ ] Implement bootstrap phase logic

### Task 1.3: Attack Implementation
- [ ] Implement untargeted attacks (random noise, sign flip)
- [ ] Implement ALIE attack
- [ ] Implement Inner Product Manipulation (IPM) attack
- [ ] Implement slow-drift attack
- [ ] Implement profile-aware adaptive attack

---

## Phase 2: Data Preparation (Weeks 2-4)

### Task 2.1: Dataset Acquisition and Preprocessing
- [ ] Obtain MIMIC-III access (requires credentialing)
- [ ] Download and preprocess ChestX-ray8
- [ ] Download and preprocess ISIC 2019
- [ ] Implement standardized preprocessing pipelines

### Task 2.2: Heterogeneity Simulation
- [ ] Implement Dirichlet allocation for label skew (α=0.5)
- [ ] Implement power law dataset size distribution
- [ ] Implement domain-specific augmentation strategies
- [ ] Create configurable client data generation

### Task 2.3: Validation Infrastructure
- [ ] Implement metrics computation (accuracy, FPR, detection latency)
- [ ] Create visualization tools for experimental results
- [ ] Implement statistical significance testing (ANOVA, t-tests)

---

## Phase 3: Experimentation (Weeks 4-8)

### Task 3.1: Baseline Experiments
- [ ] Run FedAvg without attacks (establish upper bound)
- [ ] Run all defense methods without attacks (measure false positive impact)
- [ ] Run FedAvg under each attack type (establish attack impact)
- [ ] Document all baseline results

### Task 3.2: CAAC-FL Parameter Tuning
- [ ] Tune EWMA decay parameter (α)
- [ ] Tune anomaly dimension weights (λ_mag, λ_dir, λ_temp)
- [ ] Tune reliability score parameters (γ, τ_anomaly)
- [ ] Tune bootstrap duration and transition criteria
- [ ] Document sensitivity analysis

### Task 3.3: Hypothesis Testing Experiments
- [ ] **H1 Experiments**: Compare false positive rates under heterogeneity
  - IID vs non-IID data distributions
  - Varying heterogeneity levels (Dirichlet α = 0.1, 0.5, 1.0)
  - Measure precision/recall for Byzantine detection

- [ ] **H2 Experiments**: Multi-dimensional vs single-dimension detection
  - Magnitude-only baseline
  - Direction-only baseline
  - Temporal-only baseline
  - Combined CAAC-FL
  - Test across all attack types

- [ ] **H3 Experiments**: Temporal discrimination
  - Abrupt attack onset detection
  - Slow-drift attack detection
  - Legitimate institutional evolution scenarios

### Task 3.4: Comparison Experiments
- [ ] Run full comparison matrix: 6 methods × 5 attacks × 3 datasets × 3 Byzantine fractions
- [ ] Run multiple seeds (recommend 5) for statistical validity
- [ ] Compute 95% confidence intervals

---

## Phase 4: Analysis and Writing (Weeks 8-10)

### Task 4.1: Results Analysis
- [ ] Compile all experimental results
- [ ] Perform statistical significance tests
- [ ] Create publication-quality figures and tables
- [ ] Identify key findings and surprising results

### Task 4.2: Paper Revision
- [ ] Update results section with experimental findings
- [ ] Revise claims based on actual performance
- [ ] Strengthen/qualify hypotheses based on evidence
- [ ] Address limitations section with empirical evidence
- [ ] Update related work with any new publications

### Task 4.3: Supplementary Materials
- [ ] Create appendix with full experimental details
- [ ] Prepare code repository for release
- [ ] Document hyperparameters and reproducibility instructions

---

## Phase 5: Review and Submission (Weeks 10-12)

### Task 5.1: Internal Review
- [ ] Co-author review cycle
- [ ] Address feedback and revisions
- [ ] Proofread and formatting check

### Task 5.2: Camera-Ready Preparation
- [ ] Format according to WITS guidelines
- [ ] Final proofreading
- [ ] Prepare presentation materials

---

## Critical Path Dependencies

```
Week 1-2: Infrastructure setup (parallel with data acquisition)
    ↓
Week 2-3: CAAC-FL implementation
    ↓
Week 3-4: Attack implementation + Data preprocessing
    ↓
Week 4-5: Baseline experiments + Parameter tuning
    ↓
Week 5-7: Hypothesis testing experiments
    ↓
Week 7-8: Comparison experiments
    ↓
Week 8-10: Analysis and writing
    ↓
Week 10-12: Review and submission
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| MIMIC-III access delays | Start with ChestX-ray8 and ISIC; MIMIC access application should be parallel |
| Negative experimental results | Hypotheses are testable; negative results inform future work |
| Computational resources | Use cloud computing; prioritize key experiments |
| Parameter sensitivity | Document sensitivity analysis; provide guidance for practitioners |

---

# PART IV: EXPECTED CONTRIBUTIONS TO THE LITERATURE

---

## 1. Theoretical Contribution

**Bridging Theory and Practice for Client-Specific Byzantine Defense**

Werner et al. (2023) provided theoretical proof that client-specific adaptation enhances Byzantine robustness but noted that "practical implementations of this theory remain nascent." CAAC-FL provides:

- **Concrete instantiation** of client-specific behavioral profiling
- **Algorithmic framework** that can be implemented and validated
- **Design principles** for temporal tracking in Byzantine defense

This bridges the gap between theoretical guarantees and practical deployable systems.

---

## 2. Methodological Contribution

**Multi-Dimensional Anomaly Detection Framework**

Current Byzantine defenses rely primarily on single metrics (norm, cosine similarity, or distance). CAAC-FL contributes:

- **Composite anomaly scoring** combining magnitude, direction, and temporal dimensions
- **Rationale** for why multi-dimensional detection provides stronger guarantees
- **Framework** that can be extended with additional anomaly dimensions

This provides a principled approach to designing more robust Byzantine defenses.

---

## 3. Empirical Contribution

**Systematic Evaluation in Heterogeneous Healthcare Settings**

The field lacks comprehensive evaluation of Byzantine defenses under realistic heterogeneity. Li et al. (2024) highlighted this gap. CAAC-FL will contribute:

- **First systematic comparison** of Byzantine defenses on healthcare datasets with realistic heterogeneity
- **Quantification** of the heterogeneity-robustness tradeoff
- **Evidence** on whether client-specific profiling delivers promised benefits

This provides the empirical foundation for future work in the area.

---

## 4. Practical Contribution

**Deployable Framework for Healthcare Federated Learning**

Most Byzantine defense research uses synthetic data and simplified settings. CAAC-FL contributes:

- **Healthcare-specific validation** on MIMIC-III, ChestX-ray8, and ISIC
- **Guidance on parameter selection** based on deployment context
- **Analysis of real-world challenges** (bootstrap, scalability, privacy)

This moves the field toward practical deployment in healthcare settings.

---

## 5. Paradigmatic Contribution

**Shift from Outlier Detection to Behavioral Profiling**

CAAC-FL represents a paradigm shift in how we think about Byzantine defense:

| Old Paradigm | New Paradigm (CAAC-FL) |
|--------------|------------------------|
| "Identify outliers" | "Understand individual context" |
| "Apply uniform rules" | "Apply personalized rules" |
| "Heterogeneity is a problem" | "Heterogeneity is expected" |
| "Security vs. inclusivity" | "Security through inclusivity" |

This reframing could influence future research directions in Byzantine-robust federated learning.

---

## 6. Healthcare-Specific Contribution

**Enabling Secure Multi-Institutional Medical AI**

For the healthcare informatics community, CAAC-FL contributes:

- **Framework** for secure collaboration that preserves institutional diversity
- **Recognition** that specialized centers (pediatric, geriatric, oncology) are assets, not liabilities
- **Path forward** for federated learning in realistic hospital networks

This addresses a critical barrier to adopting federated learning in healthcare practice.

---

## Summary of Expected Contributions

| Contribution Type | Description | Target Community |
|-------------------|-------------|------------------|
| Theoretical | Bridge Werner et al. theory to practice | ML Security |
| Methodological | Multi-dimensional anomaly framework | FL Research |
| Empirical | Systematic healthcare evaluation | WITS/IS |
| Practical | Deployable healthcare framework | Healthcare AI |
| Paradigmatic | Outlier detection → Behavioral profiling | Broader FL |
| Domain-Specific | Enable secure hospital collaboration | Health Informatics |

---

## Positioning Statement

CAAC-FL makes a timely contribution to the intersection of federated learning security and healthcare AI. As the 2024 systematic literature review on robust federated learning noted (Uddin et al., ACM Computing Surveys 2025), the field has extensively studied attacks and defenses but lacks solutions that work under realistic data heterogeneity.

Our work directly addresses this gap by:
1. Treating heterogeneity as a design requirement rather than an obstacle
2. Providing the first practical implementation of theoretically-motivated client-specific adaptation
3. Validating on healthcare-specific datasets with realistic heterogeneity simulation

This positions CAAC-FL as a significant step toward trustworthy federated learning for healthcare and other heterogeneous domains.

---

# APPENDIX: KEY LITERATURE REFERENCES

## Foundational Works
- Lamport, L., Shostak, R., & Pease, M. (1982). The Byzantine Generals Problem. *ACM TOPLAS*.
- McMahan, B., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS*.

## Byzantine Defense Methods
- Blanchard, P., et al. (2017). Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent. *NeurIPS*.
- Yin, D., et al. (2018). Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates. *ICML*.
- Cao, X., et al. (2021). FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping. *NDSS*.
- Pillutla, K., et al. (2021). Robust Aggregation for Federated Learning. *ICML*.

## Byzantine Attacks
- Baruch, G., Baruch, M., & Goldberg, Y. (2019). A Little Is Enough: Circumventing Defenses for Distributed Learning. *NeurIPS*.
- Fang, M., et al. (2020). Local Model Poisoning Attacks to Byzantine-Robust Federated Learning. *USENIX Security*.
- Xie, C., Koyejo, S., & Gupta, I. (2020). Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation. *UAI*.

## Recent Critical Studies
- Li, S., Ngai, E. C. H., & Voigt, T. (2024). An Experimental Study of Byzantine-Robust Aggregation Schemes in Federated Learning. *IEEE TBD*.
- Werner, M., et al. (2023). Provably Personalized and Robust Federated Learning. *arXiv:2306.08393*.

## Healthcare Federated Learning
- Rieke, N., et al. (2020). The future of digital health with federated learning. *npj Digital Medicine*.
- Dayan, I., et al. (2021). Federated learning for predicting clinical outcomes in patients with COVID-19. *Nature Medicine*.

## Recent Surveys
- Uddin, M., et al. (2025). A Systematic Literature Review of Robust Federated Learning. *ACM Computing Surveys*.
