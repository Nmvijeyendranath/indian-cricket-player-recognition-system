# Dataset Audit and Preprocessing Report

- **Total files scanned**: 7687
- **Total files excluded**: 56
- **Total clean files retained**: 7631
- **Identities represented**: 323
- **Stratification**: Train = 5215, Val = 1208, Test = 1208

## Exclusion Reasons

- `corrupt_null_bytes`: 13
- `cross_identity_conflict`: 40
- `same_identity_duplicate`: 3

## Excluded Files Detail

| File Path | Reason |
|---|---|
| `Bindeshwari Goyal\Image_8.jpg` | Cross-identity duplicate conflict between: ['Bindeshwari Goyal', 'Rajeshwari Goyal'] |
| `Dayalan Hemalatha\Image_8.jpg` | Cross-identity duplicate conflict between: ['Dayalan Hemalatha', 'Swagatika Rath'] |
| `Devika Vaidya\Image_9.jpg` | Cross-identity duplicate conflict between: ['Devika Vaidya', 'Pooja Vastrakar'] |
| `Gagan Khoda\Image_16.jpg` | Cross-identity duplicate conflict between: ['Gagan Khoda', 'Jatin Paranjpe'] |
| `Gagan Khoda\Image_19.jpg` | Cross-identity duplicate conflict between: ['Gagan Khoda', 'Jatin Paranjpe'] |
| `Gagan Khoda\Image_2.jpg` | Cross-identity duplicate conflict between: ['Gagan Khoda', 'Jatin Paranjpe'] |
| `Gagan Khoda\Image_3.jpg` | Cross-identity duplicate conflict between: ['Gagan Khoda', 'Jatin Paranjpe'] |
| `Gouher Sultana\Image_20.jpg` | Cross-identity duplicate conflict between: ['Gouher Sultana', 'Reema Malhotra'] |
| `Gundappa Viswanath\Image_10.jpg` | Cross-identity duplicate conflict between: ['Gundappa Viswanath', 'Sadanand Viswanath'] |
| `Gundappa Viswanath\Image_23.jpg` | Cross-identity duplicate conflict between: ['Gundappa Viswanath', 'Sadanand Viswanath'] |
| `Gundappa Viswanath\Image_25.jpg` | Same-identity exact duplicate of Gundappa Viswanath\Image_18.jpg |
| `Jacob Martin\Image_8.jpg` | Same-identity exact duplicate of Jacob Martin\Image_24.jpg |
| `Jatin Paranjpe\Image_18.jpg` | Cross-identity duplicate conflict between: ['Gagan Khoda', 'Jatin Paranjpe'] |
| `Jatin Paranjpe\Image_23.jpg` | Cross-identity duplicate conflict between: ['Gagan Khoda', 'Jatin Paranjpe'] |
| `Jatin Paranjpe\Image_25.jpg` | Cross-identity duplicate conflict between: ['Gagan Khoda', 'Jatin Paranjpe'] |
| `Jatin Paranjpe\Image_9.jpg` | Cross-identity duplicate conflict between: ['Gagan Khoda', 'Jatin Paranjpe'] |
| `Meghna Singh\Image_1.jpeg` | Cross-identity duplicate conflict between: ['Meghna Singh', 'Yastika Bhatia'] |
| `Mohammed Shami\Image_22.jpg` | Same-identity exact duplicate of Mohammed Shami\Image_10.jpg |
| `Mohinder Amarnath\Image_12.jpg` | Cross-identity duplicate conflict between: ['Mohinder Amarnath', 'Surinder Amarnath'] |
| `Monica Sumra\Image_17.jpg` | Cross-identity duplicate conflict between: ['Monica Sumra', 'Shubhlakshmi Sharma'] |
| `Pochiah Krishnamurthy\Image_14.jpg` | Cross-identity duplicate conflict between: ['Pochiah Krishnamurthy', 'Veda Krishnamurthy'] |
| `Pooja Vastrakar\Image_4.jpg` | Cross-identity duplicate conflict between: ['Devika Vaidya', 'Pooja Vastrakar'] |
| `Rajeshwari Goyal\Image_10.jpg` | Cross-identity duplicate conflict between: ['Bindeshwari Goyal', 'Rajeshwari Goyal'] |
| `Randhir Singh\Image_22.jpg` | Corrupt file: 100% null (0x00) bytes |
| `Randhir Singh\Image_23.jpg` | Corrupt file: 100% null (0x00) bytes |
| `Randhir Singh\Image_24.jpg` | Corrupt file: 100% null (0x00) bytes |
| `Randhir Singh\Image_25.jpg` | Corrupt file: 100% null (0x00) bytes |
| `Rashid Patel\Image_6.jpg` | Cross-identity duplicate conflict between: ['Rashid Patel', 'Vivek Razdan', 'Yastika Bhatia'] |
| `Ravi Shastri\Image_1.jpg` | Corrupt file: 100% null (0x00) bytes |
| `Ravi Shastri\Image_2.jpg` | Corrupt file: 100% null (0x00) bytes |
| `Ravi Shastri\Image_3.jpg` | Corrupt file: 100% null (0x00) bytes |
| `Ravi Shastri\Image_4.jpg` | Corrupt file: 100% null (0x00) bytes |
| `Ravi Shastri\Image_5.jpg` | Corrupt file: 100% null (0x00) bytes |
| `Ravi Shastri\Image_6.jpg` | Corrupt file: 100% null (0x00) bytes |
| `Ravi Shastri\Image_7.jpg` | Corrupt file: 100% null (0x00) bytes |
| `Ravi Shastri\Image_8.jpg` | Corrupt file: 100% null (0x00) bytes |
| `Ravi Shastri\Image_9.jpg` | Corrupt file: 100% null (0x00) bytes |
| `Reema Malhotra\Image_12.jpg` | Cross-identity duplicate conflict between: ['Gouher Sultana', 'Reema Malhotra'] |
| `Reema Malhotra\Image_15.jpg` | Cross-identity duplicate conflict between: ['Reema Malhotra', 'Simran Bahadur'] |
| `Sadanand Viswanath\Image_16.jpg` | Cross-identity duplicate conflict between: ['Gundappa Viswanath', 'Sadanand Viswanath'] |
| `Sameer Dighe\Image_22.jpg` | Cross-identity duplicate conflict between: ['Sameer Dighe', 'Syed Abid Ali'] |
| `Shahbaz Ahmed\Image_12.jpg` | Cross-identity duplicate conflict between: ['Shahbaz Ahmed', 'Shivam Dube'] |
| `Shivam Dube\Image_16.jpg` | Cross-identity duplicate conflict between: ['Shahbaz Ahmed', 'Shivam Dube'] |
| `Shubhlakshmi Sharma\Image_14.jpg` | Cross-identity duplicate conflict between: ['Monica Sumra', 'Shubhlakshmi Sharma'] |
| `Simran Bahadur\Image_4.jpg` | Cross-identity duplicate conflict between: ['Reema Malhotra', 'Simran Bahadur'] |
| `Sujith Somasunder\Image_19.jpg` | Cross-identity duplicate conflict between: ['Sujith Somasunder', 'Venkatesh Prasad'] |
| `Surinder Amarnath\Image_15.jpg` | Cross-identity duplicate conflict between: ['Surinder Amarnath', 'Surinder Khanna'] |
| `Surinder Amarnath\Image_25.jpg` | Cross-identity duplicate conflict between: ['Mohinder Amarnath', 'Surinder Amarnath'] |
| `Surinder Khanna\Image_1.jpg` | Cross-identity duplicate conflict between: ['Surinder Amarnath', 'Surinder Khanna'] |
| `Swagatika Rath\Image_11.jpg` | Cross-identity duplicate conflict between: ['Dayalan Hemalatha', 'Swagatika Rath'] |
| `Syed Abid Ali\Image_10.jpg` | Cross-identity duplicate conflict between: ['Sameer Dighe', 'Syed Abid Ali'] |
| `Veda Krishnamurthy\Image_6.jpg` | Cross-identity duplicate conflict between: ['Pochiah Krishnamurthy', 'Veda Krishnamurthy'] |
| `Venkatesh Prasad\Image_4.jpg` | Cross-identity duplicate conflict between: ['Sujith Somasunder', 'Venkatesh Prasad'] |
| `Vivek Razdan\Image_4.jpg` | Cross-identity duplicate conflict between: ['Rashid Patel', 'Vivek Razdan', 'Yastika Bhatia'] |
| `Yastika Bhatia\Image_15.jpg` | Cross-identity duplicate conflict between: ['Rashid Patel', 'Vivek Razdan', 'Yastika Bhatia'] |
| `Yastika Bhatia\Image_17.jpeg` | Cross-identity duplicate conflict between: ['Meghna Singh', 'Yastika Bhatia'] |
