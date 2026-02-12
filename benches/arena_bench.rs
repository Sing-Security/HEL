use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hel::{FactsEvalContext, Value};

#[cfg(feature = "arena")]
use hel::arena::{ArenaParser, evaluate_arena};

fn bench_parse_heap(c: &mut Criterion) {
    c.bench_function("parse_rule_heap", |b| {
        b.iter(|| {
            let ast = hel::parse_rule(black_box(r#"x == 10 AND y > 5"#));
            black_box(ast);
        });
    });
}

#[cfg(feature = "arena")]
fn bench_parse_arena(c: &mut Criterion) {
    c.bench_function("parse_rule_arena_no_reset", |b| {
        let parser = ArenaParser::new();
        b.iter(|| {
            let ast = parser.parse_rule(black_box(r#"x == 10 AND y > 5"#));
            black_box(ast);
        });
    });
}

#[cfg(feature = "arena")]
fn bench_parse_arena_with_reset(c: &mut Criterion) {
    c.bench_function("parse_rule_arena_with_reset", |b| {
        let mut parser = ArenaParser::new();
        b.iter(|| {
            let ast = parser.parse_rule(black_box(r#"x == 10 AND y > 5"#));
            black_box(ast);
            parser.reset();
        });
    });
}

fn bench_evaluate_heap(c: &mut Criterion) {
    c.bench_function("evaluate_heap", |b| {
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact("vars.x", Value::Number(10.0));
        ctx.add_fact("vars.y", Value::Number(20.0));
        
        b.iter(|| {
            let result = hel::evaluate(black_box(r#"vars.x == 10 AND vars.y > 5"#), black_box(&ctx))
                .expect("eval failed");
            black_box(result);
        });
    });
}

#[cfg(feature = "arena")]
fn bench_evaluate_arena(c: &mut Criterion) {
    c.bench_function("evaluate_arena_no_reset", |b| {
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact("vars.x", Value::Number(10.0));
        ctx.add_fact("vars.y", Value::Number(20.0));
        let parser = ArenaParser::new();
        
        b.iter(|| {
            let result = evaluate_arena(black_box(r#"vars.x == 10 AND vars.y > 5"#), black_box(&ctx), black_box(&parser))
                .expect("eval failed");
            black_box(result);
        });
    });
}

#[cfg(feature = "arena")]
fn bench_evaluate_arena_with_reset(c: &mut Criterion) {
    c.bench_function("evaluate_arena_with_reset", |b| {
        let mut ctx = FactsEvalContext::new();
        ctx.add_fact("vars.x", Value::Number(10.0));
        ctx.add_fact("vars.y", Value::Number(20.0));
        let mut parser = ArenaParser::new();
        
        b.iter(|| {
            let result = evaluate_arena(black_box(r#"vars.x == 10 AND vars.y > 5"#), black_box(&ctx), black_box(&parser))
                .expect("eval failed");
            black_box(result);
            parser.reset();
        });
    });
}

fn bench_batch_heap(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_evaluation");
    
    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("heap", size), size, |b, &size| {
            let mut ctx = FactsEvalContext::new();
            ctx.add_fact("vars.x", Value::Number(10.0));
            ctx.add_fact("vars.y", Value::Number(20.0));
            
            let expressions = vec![
                r#"vars.x == 10"#,
                r#"vars.y > 5"#,
                r#"vars.x < vars.y"#,
                r#"vars.x != 20"#,
                r#"vars.y >= 20"#,
            ];
            
            b.iter(|| {
                for _ in 0..size {
                    for expr in &expressions {
                        let result = hel::evaluate(black_box(expr), black_box(&ctx))
                            .expect("eval failed");
                        black_box(result);
                    }
                }
            });
        });
    }
    
    group.finish();
}

#[cfg(feature = "arena")]
fn bench_batch_arena(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_evaluation");
    
    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("arena", size), size, |b, &size| {
            let mut ctx = FactsEvalContext::new();
            ctx.add_fact("vars.x", Value::Number(10.0));
            ctx.add_fact("vars.y", Value::Number(20.0));
            let mut parser = ArenaParser::new();
            
            let expressions = vec![
                r#"vars.x == 10"#,
                r#"vars.y > 5"#,
                r#"vars.x < vars.y"#,
                r#"vars.x != 20"#,
                r#"vars.y >= 20"#,
            ];
            
            b.iter(|| {
                for _ in 0..size {
                    for expr in &expressions {
                        let result = evaluate_arena(black_box(expr), black_box(&ctx), black_box(&parser))
                            .expect("eval failed");
                        black_box(result);
                        parser.reset();
                    }
                }
            });
        });
    }
    
    group.finish();
}

#[cfg(feature = "arena")]
criterion_group!(
    benches,
    bench_parse_heap,
    bench_parse_arena,
    bench_parse_arena_with_reset,
    bench_evaluate_heap,
    bench_evaluate_arena,
    bench_evaluate_arena_with_reset,
    bench_batch_heap,
    bench_batch_arena
);

#[cfg(not(feature = "arena"))]
criterion_group!(
    benches,
    bench_parse_heap,
    bench_evaluate_heap,
    bench_batch_heap
);

criterion_main!(benches);
