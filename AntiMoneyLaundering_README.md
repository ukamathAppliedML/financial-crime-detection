# 🧾 Anti-Money Laundering (AML) Detection

## Domain-Specific Documentation for AML Implementation

This document provides detailed information about the **Anti-Money Laundering detection** module using the Financial Crime Detection Framework.

---

## 📊 1. AML Dataset Overview

### 1.1 Dataset Description

The AML dataset contains **726 financial communication examples** curated for training classification models to detect potential money laundering. It supports compliance surveillance and regulatory monitoring.

### 1.2 Key Statistics

- **Total Records:** 726  
- **Normal Communications:** 363 (50%)  
- **Suspicious/AML Communications:** 363 (50%)  
- **Perfect Balance:** Ensures unbiased training  

### 1.3 Dataset Structure

- **Rows:** 726  
- **Columns:** 3  
- **File Format:** CSV (UTF-8)  
- **Filename:** `AntiMoneyLaundering-v4.csv`  

---

## 📁 2. Column Descriptions

### 2.1 `label` (Integer)

Binary classification target for AML detection:

- `0`: Normal / Legitimate  
- `1`: Suspicious / AML-indicative  

### 2.2 `conversation` (String)

Contains communication content such as:

- Emails between traders and compliance  
- Internal notes on suspicious transactions  
- Client fund transfer communications  
- Discussions on reporting obligations  
- Transaction approvals and red flags  

### 2.3 `category` (String)

AML-specific subcategory for classification:

- **Normal Categories**: Standard operations  
- **Suspicious Categories**: Defined AML risk patterns (see below)  

---

## 🚨 3. AML Risk Categories: Communication-Based Red Flags

Each category includes:
- **🔍 Detection Focus**
- **✅ Suspicious Example**
- **❎ Legitimate Example**

---

### 🔹 Amount Threshold
**Detection Focus:** Avoiding $10K / $3K regulatory thresholds  
✅ _"The client wants to keep his deposits just under 10k..."_  
❎ _"The portfolio is worth over 10 million and growing..."_

---

### 🔹 Third Party Transfer  
**Detection Focus:** Use of intermediaries  
✅ _"Customer prefers using a third party... avoids his name..."_  
❎ _"The wire transfer requires third-party compliance approval..."_

---

### 🔹 AML Concerns  
**Detection Focus:** Direct mention of AML red flags  
✅ _"Compliance flagged several transactions for AML review..."_  
❎ _"We need to update our AML procedures to comply with new rules..."_

---

### 🔹 Tax Avoidance  
**Detection Focus:** Evasion of tax obligations  
✅ _"Client asked about avoiding tax on offshore returns..."_  
❎ _"Analyze tax implications and coordinate with tax advisor..."_

---

### 🔹 CTR Issues  
**Detection Focus:** Problems with CTR filing  
✅ _"The CTR for 18k deposit was never filed..."_  
❎ _"CTR was filed electronically before the deadline..."_

---

### 🔹 Offshore Accounts  
**Detection Focus:** Obscured offshore sources  
✅ _"Funds from Cayman Islands; vague business purpose..."_  
❎ _"Well-documented Singapore business operations..."_

---

### 🔹 Structuring  
**Detection Focus:** Breaking up deposits  
✅ _"Client deposits just under 10k over different days..."_  
❎ _"Optimized asset allocation; legal investment structure..."_

---

### 🔹 Smurfing  
**Detection Focus:** Multiple coordinated small deposits  
✅ _"Classic smurfing pattern with multiple accounts..."_  
❎ _"Regular small deposits from retail customers..."_

---

### 🔹 Layering  
**Detection Focus:** Complex transfers across accounts  
✅ _"Funds move through multiple jurisdictions..."_  
❎ _"Layered diversification for investment risk management..."_

---

### 🔹 Micro-structuring  
**Detection Focus:** Hundreds of tiny deposits  
✅ _"Client uses micro-structuring under $1000 thresholds..."_  
❎ _"Analyzing micro-transactions for efficiency..."_

---

### 🔹 Clean Money  
**Detection Focus:** Explicit mention of laundering  
✅ _"Client wants to clean his money through investments..."_  
❎ _"Clean up documentation for compliance audit..."_

---

### 🔹 Shell Company  
**Detection Focus:** Entities without real operations  
✅ _"Funneling funds through shell company..."_  
❎ _"Registered entity with corporate governance..."_

---

### 🔹 Beneficial Ownership  
**Detection Focus:** Obscuring ownership  
✅ _"Client is evasive about account owners..."_  
❎ _"Ownership documentation is complete and verified..."_

---

### 🔹 Geographic Risk  
**Detection Focus:** High-risk or sanctioned countries  
✅ _"Business interests in North Korea raise sanctions concerns..."_  
❎ _"Global operations with verified foreign registrations..."_  
