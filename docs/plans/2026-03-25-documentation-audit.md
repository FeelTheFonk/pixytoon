# ULTIMATE Documentation Audit & Refactoring Implementation Plan

> **For Antigravity:** REQUIRED SUB-SKILL: Load executing-plans to implement this plan task-by-task.

**Goal:** Une refonte SOTA absolue, stricte et méticuleuse de l'intégralité de la suite documentaire (README, guides, architecture, entrypoints) pour atteindre la perfection (100/100). Cohérence technique, beauté visuelle (Mermaid natif), suppression du bruit ASCII, et correction matérielle.

**Architecture de la refonte:**
1. **README.md** deviendra une vitrine pure, concise, avec avertissements OS/GPU corrigés.
2. **ARCHITECTURE.md** absorbera la complexité technique et l'arbre de fichiers.
3. **AUDIO-REACTIVITY.md** sera purgé de son ASCII pour un Flowchart Mermaid SOTA.
4. **CONTRIBUTING.md & GUIDE.md** seront alignés techniquement (entrypoints corrigés, VRAM ajusté).

**Tech Stack:** Markdown, Mermaid, Regex.

---

### Task 1: Dépollution et Vente du README.md (Correction VRAM & OS)

**Files:**
- Modify: `README.md`

**Step 1: Write the failing test**
N/A (Documentation)

**Step 2: Run test to verify it fails**
N/A

**Step 3: Write minimal implementation**
- **Requirements** : Ligne 207, modifier pour refléter la stricte vérité : `NVIDIA >= 4GB VRAM (txt2img/audio at 512x512). 8GB+ recommended for AnimateDiff/ControlNet.` Ajouter une ligne claire : `OS: Windows 10/11 only (requires PowerShell 7)`.
- **Architecture Diagram** : Lignes 20-37, supprimer l'ASCII affreux et remplacer par : `See **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** for detailed Mermaid diagrams of the system design.`
- **Project Structure** : Lignes 38-119, purger l'immense arbre ASCII. Le remplacer par un résumé de 4 lignes de l'architecture backend/frontend. Le vieil arbre sera déplacé vers `ARCHITECTURE.md`.
- **Attention Mechanism** : Lignes 194-204, couper intégralement ce bloc technique et le préparer pour le déplacement vers `ARCHITECTURE.md`.
- **Features (Audio)** : Condenser la mention des 34 paramètres audio (lignes 152) pour renvoyer vers le fichier audio.

**Step 4: Run test to verify it passes**
- Check markdown preview rendering.

**Step 5: Commit**
```bash
git add README.md
git commit -m "docs: ultimate README cleanup - correct VRAM to 4GB, add Windows warning, purge ASCII"
```

---

### Task 2: Synchronisation SOTA de ARCHITECTURE.md

**Files:**
- Modify: `docs/ARCHITECTURE.md`

**Step 1: Write the failing test**
N/A

**Step 2: Run test to verify it fails**
N/A

**Step 3: Write minimal implementation**
- Insérer la section `Attention Mechanism` (récupérée de README) sous la division `Diffusion Pipeline`.
- Créer une section `Project Structure` à la fin du fichier et y coller l'arborescence de fichiers complète (qui avait été purgée du README), permettant aux développeurs d'avoir cette information complexe sans polluer l'interface utilisateur.

**Step 4: Run test to verify it passes**
- Verify markdown rendering.

**Step 5: Commit**
```bash
git add docs/ARCHITECTURE.md
git commit -m "docs(arch): ingest technical depth (file tree and attention mechanism) from README"
```

---

### Task 3: Beauté Visuelle de AUDIO-REACTIVITY.md (Mermaid SOTA)

**Files:**
- Modify: `docs/AUDIO-REACTIVITY.md`

**Step 1: Write the failing test**
N/A

**Step 2: Run test to verify it fails**
N/A

**Step 3: Write minimal implementation**
- Lignes 15-52 : Supprimer le bloc ASCII d'architecture.
- Créer un bloc ````mermaid` contenant un `graph TD` élégant et sémantiquement parfait, reproduisant le pipeline DSP avec des noeuds : Audio File -> Analyzer -> Modulation Engine -> Parameter Schedule -> (Frame Chain / AnimateDiff) -> Post-Processing -> MP4 Export.

**Step 4: Run test to verify it passes**
- Verify the mermaid code successfully renders in standard Github markdown parsers.

**Step 5: Commit**
```bash
git add docs/AUDIO-REACTIVITY.md
git commit -m "docs(audio): replace legacy ASCII pipeline flow with SOTA Mermaid graph"
```

---

### Task 4: Cohérence du GUIDE.md (Correction Matérielle Intégrale)

**Files:**
- Modify: `docs/GUIDE.md`

**Step 1: Write the failing test**
N/A

**Step 2: Run test to verify it fails**
N/A

**Step 3: Write minimal implementation**
- Ligne 36 (Prerequisites) : Remplacer "NVIDIA GPU with at least 8 GB VRAM" par "NVIDIA GPU with at least 4 GB VRAM (txt2img/audio at 512x512). 8GB+ for AnimateDiff/ControlNet."
- Lignes 606-614 (VRAM usage table) : Ajuster les valeurs. `Idle` passe à `~3-4 GB`. `Generate 512x512` passe à `~4-5 GB`.
- Ajouter une note sur `HF_HUB_OFFLINE=1` : Expliquer sous "First Launch" que le démarrage local désactive toute connectivité cloud/télémétrie par défaut, garantissant le respect de la vie privée (un argument de vente majeur omis).

**Step 4: Run test to verify it passes**
- Visual test of markdown tables.

**Step 5: Commit**
```bash
git add docs/GUIDE.md
git commit -m "docs(guide): standardize 4GB VRAM truth and emphasize 100% offline privacy"
```

---

### Task 5: Rigueur Développeur sur CONTRIBUTING.md

**Files:**
- Modify: `CONTRIBUTING.md`

**Step 1: Write the failing test**
N/A

**Step 2: Run test to verify it fails**
N/A

**Step 3: Write minimal implementation**
- Ligne 21 : Remplacer `uv run sddj-server --reload` par la commande d'entrée vérifiée dans `start.ps1`, c'est-à-dire `uv run python run.py`. Cette asymétrie faisait perdre un temps précieux aux contributeurs.

**Step 4: Run test to verify it passes**
- Verify file content.

**Step 5: Commit**
```bash
git add CONTRIBUTING.md
git commit -m "docs(contrib): fix backend run command mismatch to align with start.ps1"
```
