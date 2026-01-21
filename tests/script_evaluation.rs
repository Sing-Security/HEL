//! Integration tests for HEL script evaluation with let bindings
//!
//! These tests demonstrate end-to-end script evaluation workflows.

use hel::{evaluate_script, parse_script, validate_expression, FactsEvalContext, Value};

#[test]
fn test_android_malware_detection_script() {
    // Simulate Android app analysis
    let mut ctx = FactsEvalContext::new();
    
    // Binary characteristics
    ctx.add_fact("binary.arch", Value::String("arm".into()));
    ctx.add_fact("binary.entropy", Value::Number(8.2));
    ctx.add_fact("strings.count", Value::Number(5.0));
    
    // Manifest permissions
    ctx.add_fact("manifest.permissions", Value::List(vec![
        Value::String("READ_SMS".into()),
        Value::String("SEND_SMS".into()),
        Value::String("INTERNET".into()),
        Value::String("READ_CONTACTS".into()),
    ]));
    
    let script = r#"
        # Check for suspicious SMS permissions
        let has_sms_perms = 
          manifest.permissions CONTAINS "READ_SMS" AND
          manifest.permissions CONTAINS "SEND_SMS"
        
        # Check for code obfuscation indicators
        let has_obfuscation = 
          binary.entropy > 7.5 OR
          strings.count < 10
        
        # Final detection logic
        has_sms_perms AND has_obfuscation
    "#;
    
    let result = evaluate_script(script, &ctx).expect("evaluation failed");
    assert!(result, "Should detect suspicious app");
}

#[test]
fn test_binary_security_analysis_script() {
    let mut ctx = FactsEvalContext::new();
    
    ctx.add_fact("binary.format", Value::String("ELF".into()));
    ctx.add_fact("binary.arch", Value::String("x86_64".into()));
    ctx.add_fact("security.nx", Value::Bool(true));
    ctx.add_fact("security.pie", Value::Bool(true));
    ctx.add_fact("security.relro", Value::Bool(true));
    ctx.add_fact("security.stack_canary", Value::Bool(true));
    
    let script = r#"
        # Modern security features
        let has_modern_protections = 
          security.nx == true AND
          security.pie == true AND
          security.stack_canary == true
        
        # Architecture check
        let is_64bit = binary.arch == "x86_64"
        
        # Binary is secure if it's 64-bit with modern protections
        is_64bit AND has_modern_protections
    "#;
    
    let result = evaluate_script(script, &ctx).expect("evaluation failed");
    assert!(result, "Binary should be considered secure");
}

#[test]
fn test_network_behavior_analysis() {
    let mut ctx = FactsEvalContext::new();
    
    ctx.add_fact("network.domains", Value::List(vec![
        Value::String("api.example.com".into()),
        Value::String("malicious.cc".into()),
        Value::String("c2server.xyz".into()),
    ]));
    
    ctx.add_fact("network.port_count", Value::Number(15.0));
    ctx.add_fact("network.unique_ips", Value::Number(23.0));
    
    let script = r#"
        # Check for C2 infrastructure indicators
        let has_c2_domains = 
          network.domains CONTAINS "c2server.xyz" OR
          network.domains CONTAINS "malicious.cc"
        
        # Check for port scanning behavior
        let shows_scanning = network.port_count > 10
        
        # Check for botnet behavior
        let shows_botnet = network.unique_ips > 20
        
        # Detection logic
        has_c2_domains AND (shows_scanning OR shows_botnet)
    "#;
    
    let result = evaluate_script(script, &ctx).expect("evaluation failed");
    assert!(result, "Should detect malicious network behavior");
}

#[test]
fn test_script_validation_catches_errors() {
    // Truly incomplete expression - missing closing parenthesis
    let invalid_script = r#"
        let has_perms = (binary.format == "ELF"
        has_perms
    "#;
    
    let result = parse_script(invalid_script);
    assert!(result.is_err(), "Should catch incomplete expression");
}

#[test]
fn test_complex_conditional_logic() {
    let mut ctx = FactsEvalContext::new();
    
    ctx.add_fact("file.type", Value::String("PE".into()));
    ctx.add_fact("file.size", Value::Number(1024000.0));
    ctx.add_fact("signatures.matched", Value::List(vec![
        Value::String("packed".into()),
        Value::String("encrypted".into()),
    ]));
    ctx.add_fact("behavior.creates_files", Value::Bool(true));
    ctx.add_fact("behavior.modifies_registry", Value::Bool(true));
    
    let script = r#"
        # Packer detection
        let is_packed = signatures.matched CONTAINS "packed"
        
        # Encryption detection
        let is_encrypted = signatures.matched CONTAINS "encrypted"
        
        # Suspicious behavior
        let suspicious_behavior = 
          behavior.creates_files == true AND
          behavior.modifies_registry == true
        
        # Large file with obfuscation
        let large_and_obfuscated = 
          file.size > 1000000 AND
          (is_packed OR is_encrypted)
        
        # Final verdict
        large_and_obfuscated AND suspicious_behavior
    "#;
    
    let result = evaluate_script(script, &ctx).expect("evaluation failed");
    assert!(result, "Should flag suspicious executable");
}

#[test]
fn test_chained_let_bindings() {
    let mut ctx = FactsEvalContext::new();
    
    ctx.add_fact("data.value", Value::Number(100.0));
    ctx.add_fact("data.threshold", Value::Number(50.0));
    
    let script = r#"
        let exceeds_threshold = data.value > data.threshold
        let double_threshold = data.value > 100
        let condition_met = exceeds_threshold AND double_threshold == false
        condition_met
    "#;
    
    let result = evaluate_script(script, &ctx).expect("evaluation failed");
    assert!(result, "Chained bindings should work correctly");
}

#[test]
fn test_empty_list_handling() {
    let mut ctx = FactsEvalContext::new();
    
    ctx.add_fact("data.items", Value::List(vec![]));
    ctx.add_fact("data.has_data", Value::Bool(false));
    
    let script = r#"
        let list_is_empty = data.has_data == false
        list_is_empty
    "#;
    
    let result = evaluate_script(script, &ctx).expect("evaluation failed");
    assert!(result, "Should handle empty lists");
}

#[test]
fn test_numeric_comparisons_in_bindings() {
    let mut ctx = FactsEvalContext::new();
    
    ctx.add_fact("score.risk", Value::Number(7.8));
    ctx.add_fact("score.confidence", Value::Number(0.92));
    
    let script = r#"
        let high_risk = score.risk > 7.0
        let high_confidence = score.confidence > 0.9
        let should_alert = high_risk AND high_confidence
        should_alert
    "#;
    
    let result = evaluate_script(script, &ctx).expect("evaluation failed");
    assert!(result, "Numeric comparisons should work in bindings");
}

#[test]
fn test_script_with_comments_only() {
    let ctx = FactsEvalContext::new();
    
    let script = r#"
        # This is just a comment
        # Another comment
        true
    "#;
    
    let result = evaluate_script(script, &ctx).expect("evaluation failed");
    assert!(result, "Should handle scripts with comments");
}

#[test]
fn test_script_parsing_validation() {
    // Valid script should parse
    let valid = r#"
        let x = true
        x
    "#;
    assert!(parse_script(valid).is_ok());
    
    // Script must have final expression
    let no_final = r#"
        let x = true
    "#;
    assert!(parse_script(no_final).is_err());
}
