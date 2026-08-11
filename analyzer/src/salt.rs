use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

use ahash::AHashMap;
use denning::MissRatioCurve;
use raffine::{
    affine::{AffineExpr, AffineMap},
    tree::{Tree, ValID},
};
use serde::Serialize;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        Field, RingOps,
        integer::{Integer, IntegerRing},
        rational_polynomial::{FromNumeratorAndDenominator, RationalPolynomialField},
    },
    printer::PrintOptions,
    symbol,
};

use crate::{
    AnalysisContext,
    utils::{Poly, convert_affine_map},
};

const ROOT_REUSE_FACTOR: usize = usize::MAX;

fn constant_poly(value: isize, context: &AnalysisContext<'_>) -> Poly {
    let expr = AffineExpr::new_constant(context.rcontext().mlir_context(), value as i64);
    let map = AffineMap::new(
        context.rcontext().mlir_context(),
        0, // num_dims
        0, // num_symbols
        &[expr],
    );
    convert_affine_map(map, &[])
        .expect("a constant affine map must be convertible")
        .into_iter()
        .next()
        .expect("the affine map has exactly one result")
}

fn trip_count_to_poly(id: usize) -> Poly {
    let ring = IntegerRing::new();
    let atom = Atom::var(symbol!(format!("tc{id}")));
    atom.to_rational_polynomial(&ring, &ring, None)
}

fn iteration_variable_index(variable: &impl ToString) -> Option<usize> {
    variable.to_string().strip_prefix('i')?.parse().ok()
}

/// Returns whether every induction-variable coefficient in every access map is one.
///
/// SALT's block model assumes that loop indices advance accessed locations with unit
/// coefficients. Trees containing conditionals or unconvertible affine maps are rejected.
#[allow(
    dead_code,
    reason = "available as an optional SALT applicability check"
)]
pub fn no_coefficient_for_block(tree: &Tree<'_>) -> bool {
    match tree {
        Tree::For { body, .. } => no_coefficient_for_block(body),
        Tree::Access { map, operands, .. } => {
            convert_affine_map(*map, operands).is_ok_and(|polys| {
                polys.iter().all(|poly| {
                    poly.numerator
                        .variables
                        .iter()
                        .zip(poly.numerator.coefficients.iter())
                        .filter(|(variable, _)| iteration_variable_index(variable).is_some())
                        .all(|(_, coefficient)| *coefficient == 1)
                })
            })
        }
        Tree::Block(trees) => {
            !trees.is_empty() && trees.iter().all(|tree| no_coefficient_for_block(tree))
        }
        Tree::If { .. } => false,
    }
}

#[allow(dead_code, reason = "supports the optional SALT reuse check")]
fn has_reuses_helper(tree: &Tree<'_>, active_ivars: &mut HashSet<usize>) -> bool {
    match tree {
        Tree::For { body, ivar, .. } => {
            let ValID::IVar(id) = ivar else {
                return false;
            };

            let inserted = active_ivars.insert(*id);
            let has_reuses = has_reuses_helper(body, active_ivars);
            if inserted {
                active_ivars.remove(id);
            }
            has_reuses
        }
        Tree::Access { map, operands, .. } => {
            let Ok(polys) = convert_affine_map(*map, operands) else {
                return false;
            };
            let referenced_ivars = polys
                .iter()
                .flat_map(|poly| poly.numerator.variables.iter())
                .filter_map(iteration_variable_index)
                .collect::<HashSet<_>>();

            active_ivars.iter().any(|id| !referenced_ivars.contains(id))
        }
        Tree::Block(trees) => {
            !trees.is_empty()
                && trees
                    .iter()
                    .all(|tree| has_reuses_helper(tree, active_ivars))
        }
        Tree::If { .. } => false,
    }
}

/// Returns whether every access reuses data along at least one enclosing loop dimension.
#[allow(
    dead_code,
    reason = "available as an optional SALT applicability check"
)]
pub fn has_reuses(tree: &Tree<'_>) -> bool {
    has_reuses_helper(tree, &mut HashSet::new())
}

/// Returns whether the tree is a single loop nest ending in one or more accesses.
#[allow(
    dead_code,
    reason = "available as an optional SALT applicability check"
)]
pub fn is_perfectly_nested(tree: &Tree<'_>) -> bool {
    match tree {
        Tree::For { body, .. } => is_perfectly_nested(body),
        Tree::Block([]) => false,
        Tree::Block([tree]) => is_perfectly_nested(tree),
        Tree::Block(trees) => trees.iter().all(|tree| matches!(tree, Tree::Access { .. })),
        Tree::Access { .. } => true,
        Tree::If { .. } => false,
    }
}

/// Counts the memory accesses represented by the loop tree.
pub fn number_of_accesses(tree: &Tree) -> usize {
    match tree {
        Tree::For { body, .. } => number_of_accesses(body),
        Tree::Block(trees) => trees
            .iter()
            .filter(|subtree| matches!(subtree, Tree::Access { .. }))
            .count(),
        Tree::Access { .. } => 1,
        Tree::If { .. } => 0,
    }
}

/// Computes the reuse-interval distribution for an unblocked loop tree.
pub fn get_reuse_interval_distribution<'a, 'b: 'a>(
    tree: &Tree<'a>,
    reuse_factors: &mut HashMap<usize, Poly>,
    trip_counts: &mut HashMap<usize, Poly>,
    ref_count: usize,
    context: &AnalysisContext<'b>,
) -> HashMap<Poly, Poly> {
    match tree {
        Tree::For {
            lower_bound,
            upper_bound,
            lower_bound_operands,
            upper_bound_operands,
            body,
            ivar,
            step,
        } => {
            let lower_bound_converted = convert_affine_map(*lower_bound, lower_bound_operands)
                .expect("a loop lower bound must be convertible");
            let upper_bound_converted = convert_affine_map(*upper_bound, upper_bound_operands)
                .expect("a loop upper bound must be convertible");
            let ValID::IVar(id) = ivar else {
                unreachable!("a loop induction variable must be an IVar")
            };
            let field = RationalPolynomialField::new(IntegerRing);
            let iteration_span =
                upper_bound_converted[0].clone() - lower_bound_converted[0].clone();
            let raw_trip_count = field.div(&iteration_span, &constant_poly(*step, context));
            let trip_count = if raw_trip_count.get_variables().is_empty() {
                raw_trip_count
            } else {
                trip_count_to_poly(*id)
            };
            if reuse_factors.is_empty() {
                reuse_factors.insert(ROOT_REUSE_FACTOR, constant_poly(1, context));
            }

            for value in reuse_factors.values_mut() {
                *value = &*value * &trip_count;
            }

            reuse_factors.insert(*id, constant_poly(1, context));
            trip_counts.insert(*id, trip_count.clone());

            get_reuse_interval_distribution(body, reuse_factors, trip_counts, ref_count, context)
        }
        Tree::Block(trees) => {
            let mut ri_dist: HashMap<Poly, Poly> = HashMap::new();
            for subtree in trees.iter() {
                let subtree_distribution = get_reuse_interval_distribution(
                    subtree,
                    reuse_factors,
                    trip_counts,
                    trees.len(),
                    context,
                );
                for (interval, portion) in subtree_distribution {
                    ri_dist
                        .entry(interval)
                        .and_modify(|current| *current = &*current + &portion)
                        .or_insert(portion);
                }
            }
            ri_dist
        }
        Tree::Access { map, operands, .. } => {
            let field = RationalPolynomialField::new(IntegerRing);

            // Record the loop dimensions used by this access and their trip-count product.
            let mut reference_vector = vec![0; reuse_factors.len()];
            let mut block_position = 0;
            let mut referenced_trip_product = constant_poly(1, context);
            if let Ok(polys) = convert_affine_map(*map, operands) {
                for poly in polys {
                    block_position = 0;
                    for var in poly.numerator.variables.iter() {
                        if let Some(index) = iteration_variable_index(var) {
                            block_position = block_position.max(index + 1);
                            reference_vector[index + 1] = 1;
                            referenced_trip_product = &referenced_trip_product
                                * trip_counts
                                    .get(&index)
                                    .expect("each referenced loop must have a trip count");
                        }
                    }
                }
            }
            reference_vector[block_position] = 1;

            // Compute the share of accesses associated with each loop dimension.
            let mut portion_factors = vec![];
            let mut portion_denominator = constant_poly(1, context);
            let mut total_access = constant_poly(1, context);
            for i in (0..reference_vector.len()).rev() {
                portion_factors.push(field.div(&constant_poly(1, context), &portion_denominator));
                if reference_vector[i] == 0 && i != 0 {
                    portion_denominator = &portion_denominator
                        * trip_counts
                            .get(&(i - 1))
                            .expect("each enclosing loop must have a trip count");
                }
                if i != 0 {
                    total_access = &total_access
                        * trip_counts
                            .get(&(i - 1))
                            .expect("each enclosing loop must have a trip count");
                }
            }
            portion_factors.reverse();

            reference_vector[block_position] = 2;

            // Keep only transitions between referenced and unreferenced dimensions.
            let mut reference_boundaries = vec![];
            let mut coefficients = vec![];
            let mut passed_block_position = false;
            let mut last_position_of_zero_group = reference_vector.len() - 1;

            for i in (0..reference_vector.len() - 1).rev() {
                if reference_vector[i] != reference_vector[i + 1] {
                    if i == block_position {
                        if reference_vector[i + 1] == 1 {
                            coefficients.push(0);
                            last_position_of_zero_group = block_position;
                        } else {
                            reference_vector[i] = 1;
                            passed_block_position = true;
                            continue;
                        }
                    } else if reference_vector[i] == 1 {
                        coefficients.push(-1);
                    } else if reference_vector[i] == 0 {
                        coefficients.push(1);
                        if !passed_block_position {
                            last_position_of_zero_group = i;
                        }
                    }
                    reference_boundaries.push(i);
                }
                if i == block_position {
                    passed_block_position = true;
                }
            }

            reference_boundaries.reverse();
            coefficients.reverse();
            if reference_vector[reference_vector.len() - 1] == 0 {
                reference_boundaries.push(reference_vector.len() - 1);
                coefficients.push(1);
            } else if reference_vector[reference_vector.len() - 1] == 2 {
                reference_boundaries.push(reference_vector.len() - 1);
                coefficients.push(0);
                last_position_of_zero_group = reference_vector.len() - 1;
            }

            let reference_count = constant_poly(ref_count as isize, context);
            let mut ri_value = constant_poly(0, context);

            let block_atom = Atom::var(symbol!("b"));
            let block_poly =
                block_atom.to_rational_polynomial(&IntegerRing::new(), &IntegerRing::new(), None);
            let mut reuse_intervals: Vec<(Poly, usize)> = vec![];

            let mut block_interval = constant_poly(-1, context);

            for (place, position) in reference_boundaries.iter().rev().enumerate() {
                let factor = if *position != 0 {
                    reuse_factors
                        .get(&(*position - 1))
                        .expect("each enclosing loop must have a reuse factor")
                } else {
                    reuse_factors
                        .get(&ROOT_REUSE_FACTOR)
                        .expect("the root reuse factor must be initialized")
                };

                let coefficient = coefficients[coefficients.len() - 1 - place];

                if last_position_of_zero_group == *position {
                    ri_value = &ri_value + factor;
                    reuse_intervals.push(((&ri_value * &reference_count), *position));

                    block_interval = &ri_value * &reference_count;
                    let block_factor = if block_position != 0 {
                        reuse_factors
                            .get(&(block_position - 1))
                            .expect("the block loop must have a reuse factor")
                    } else {
                        reuse_factors
                            .get(&ROOT_REUSE_FACTOR)
                            .expect("the root reuse factor must be initialized")
                    };
                    ri_value = &ri_value - &(&block_poly * block_factor);
                } else if coefficient == -1 {
                    if reuse_intervals.len() > 1 {
                        ri_value = &ri_value - factor;
                    }
                } else {
                    ri_value = &ri_value + factor;
                    reuse_intervals.push(((&ri_value * &reference_count), *position));
                }
            }

            reuse_intervals.reverse();

            let mut ri_dist: HashMap<Poly, Poly> = HashMap::new();

            let mut previous_portion = constant_poly(0, context);

            let mut block_portion_adjustment = constant_poly(0, context);
            let mut saw_first_interval = false;
            for (interval, position) in &reuse_intervals {
                let portion_factor = portion_factors[*position].clone();
                if saw_first_interval {
                    if *position < block_position {
                        let without_block = &portion_factor - &previous_portion;
                        let with_block = field.div(&without_block, &block_poly);
                        block_portion_adjustment =
                            &block_portion_adjustment + &(&without_block - &with_block);
                        ri_dist.insert(interval.clone(), field.div(&with_block, &reference_count));
                    } else {
                        let portion = &portion_factor - &previous_portion;
                        ri_dist.insert(interval.clone(), field.div(&portion, &reference_count));
                    }
                    previous_portion = portion_factor;
                } else {
                    saw_first_interval = true;
                    let imaginary_portion = field.div(
                        &field.div(
                            &field.div(&referenced_trip_product, &total_access),
                            &reference_count,
                        ),
                        &block_poly,
                    );
                    if *position < block_position {
                        let without_block = &portion_factor - &previous_portion;
                        let with_block = field.div(&without_block, &block_poly);
                        block_portion_adjustment =
                            &block_portion_adjustment + &(&without_block - &with_block);
                        let portion = field.div(&with_block, &reference_count);
                        if portion != constant_poly(0, context) {
                            ri_dist.insert(interval.clone(), portion);
                        }
                    } else {
                        let portion = &portion_factor - &previous_portion;
                        let portion = &field.div(&portion, &reference_count) - &imaginary_portion;
                        if portion != constant_poly(0, context) {
                            ri_dist.insert(interval.clone(), portion);
                        }
                    }
                    previous_portion = portion_factor;
                }
            }

            let block_portion = ri_dist
                .get_mut(&block_interval)
                .expect("the block reuse interval must be present");
            *block_portion =
                &*block_portion + &field.div(&block_portion_adjustment, &reference_count);
            ri_dist
        }

        Tree::If { .. } => HashMap::new(),
    }
}

#[derive(Serialize)]
struct SaltResult {
    ri_values: Vec<String>,
    portions: Vec<String>,
    total_count: String,
    miss_ratio_curve: MissRatioCurve,
    analysis_time: Duration,
}

/// Multiplies the access count by every enclosing loop's trip count.
pub fn get_total_count<'a, I>(accesses: usize, trip_counts: I) -> Poly
where
    I: Iterator<Item = &'a Poly>,
{
    let ring = IntegerRing::new();
    let field = RationalPolynomialField::new(ring);
    let accesses = Atom::num(accesses as i64).to_rational_polynomial(&ring, &ring, None);
    trip_counts.fold(accesses, |acc, poly| field.mul(&acc, poly))
}

/// Converts the numeric portion of a symbolic distribution into curve input.
///
/// Entries that still contain symbolic values are omitted.
pub fn get_ri_distribution(dist: &[(Poly, Poly)]) -> Vec<(isize, f64)> {
    let mut distro_map = AHashMap::new();
    let empty_const_map = AHashMap::<Atom, _>::new();
    let empty_symbol_map = AHashMap::new();
    let mut saw_numeric_value = false;
    for (value, portion) in dist.iter() {
        let Ok(value) =
            value
                .to_expression()
                .evaluate(|x| x.to_f64(), &empty_const_map, &empty_symbol_map)
        else {
            continue;
        };
        let Ok(portion) =
            portion
                .to_expression()
                .evaluate(|x| x.to_f64(), &empty_const_map, &empty_symbol_map)
        else {
            continue;
        };
        saw_numeric_value = true;
        let value = value as isize;
        distro_map
            .entry(value)
            .and_modify(|total| *total += portion)
            .or_insert(portion);
    }
    if !saw_numeric_value {
        return Vec::new();
    }
    let mut distro = distro_map.into_iter().chain([(0, 0.0)]).collect::<Vec<_>>();
    distro.sort_by_key(|entry| entry.0);
    distro
}

/// Replaces the symbolic block-size variable `b` with a concrete value.
pub fn substitute_block_size(poly: &Poly, block_size: usize) -> Poly {
    let vars = poly.get_variables();
    let var_idx = vars.iter().position(|v| {
        v.get_id()
            .map(|id| id.get_stripped_name() == "b")
            .unwrap_or(false)
    });
    match var_idx {
        Some(idx) => {
            let ring = IntegerRing::new();
            let integer = Integer::new(block_size as i64);
            let numerator = poly.numerator.replace(idx, &integer);
            let denominator = poly.denominator.replace(idx, &integer);
            Poly::from_num_den(numerator, denominator, &ring, true)
        }
        None => poly.clone(),
    }
}

/// Serializes the symbolic distribution and its derived miss-ratio curve.
pub fn create_json_output<'a, I>(
    dist: &[(Poly, Poly)],
    accesses: usize,
    trip_counts: I,
    start_time: Instant,
) -> anyhow::Result<String>
where
    I: Iterator<Item = &'a Poly>,
{
    let total_count = get_total_count(accesses, trip_counts);
    let ri_values: Vec<String> = dist
        .iter()
        .map(|(poly, _)| {
            poly.to_expression()
                .printer(PrintOptions::file_no_namespace())
                // .printer(PrintOptions::latex())
                .to_string()
        })
        .collect();
    let portions: Vec<String> = dist
        .iter()
        .map(|(_, poly)| {
            poly.to_expression()
                .printer(PrintOptions::file_no_namespace())
                .to_string()
        })
        .collect();
    let total_count = total_count
        .to_expression()
        .printer(PrintOptions::file_no_namespace())
        .to_string();
    let distribution = get_ri_distribution(dist).into_boxed_slice();
    let miss_ratio_curve = MissRatioCurve::new(&distribution);
    let analysis_time = start_time.elapsed();
    let result = SaltResult {
        ri_values,
        portions,
        total_count,
        miss_ratio_curve,
        analysis_time,
    };
    serde_json::to_string(&result).map_err(|e| anyhow::anyhow!("Failed to serialize to JSON: {e}"))
}

#[cfg(test)]
mod tests {
    use raffine::{Context, affine::AffineExpr};

    use super::*;

    fn access<'a>(map: AffineMap<'a>, operands: &'a [ValID]) -> Tree<'a> {
        Tree::Access {
            memref: ValID::Memref(0),
            map,
            operands,
            is_write: false,
        }
    }

    #[test]
    fn coefficient_check_inspects_every_map_result() {
        let context = Context::new();
        let mlir_context = context.mlir_context();
        let bad = AffineExpr::new_dim(mlir_context, 0) * AffineExpr::new_constant(mlir_context, 2);
        let good = AffineExpr::new_dim(mlir_context, 0);
        let map = AffineMap::new(mlir_context, 1, 0, &[bad, good]);
        let operands = [ValID::IVar(0)];
        let tree = access(map, &operands);

        assert!(!no_coefficient_for_block(&tree));
    }

    #[test]
    fn reuse_check_does_not_leak_ivars_between_sibling_loops() {
        let context = Context::new();
        let mlir_context = context.mlir_context();
        let lower_bound = AffineMap::new_constant(mlir_context, 0);
        let upper_bound = AffineMap::new_constant(mlir_context, 10);

        let constant_access = access(AffineMap::new_constant(mlir_context, 0), &[]);
        let first_loop = Tree::For {
            lower_bound,
            upper_bound,
            lower_bound_operands: &[],
            upper_bound_operands: &[],
            step: 1,
            ivar: ValID::IVar(0),
            body: &constant_access,
        };

        let second_operands = [ValID::IVar(1)];
        let second_access = access(
            AffineMap::new(mlir_context, 1, 0, &[AffineExpr::new_dim(mlir_context, 0)]),
            &second_operands,
        );
        let second_loop = Tree::For {
            lower_bound,
            upper_bound,
            lower_bound_operands: &[],
            upper_bound_operands: &[],
            step: 1,
            ivar: ValID::IVar(1),
            body: &second_access,
        };
        let siblings = [&first_loop, &second_loop];
        let tree = Tree::Block(&siblings);

        assert!(!has_reuses(&tree));
    }

    #[test]
    fn reuse_check_detects_an_omitted_enclosing_ivar() {
        let context = Context::new();
        let mlir_context = context.mlir_context();
        let constant_access = access(AffineMap::new_constant(mlir_context, 0), &[]);
        let tree = Tree::For {
            lower_bound: AffineMap::new_constant(mlir_context, 0),
            upper_bound: AffineMap::new_constant(mlir_context, 10),
            lower_bound_operands: &[],
            upper_bound_operands: &[],
            step: 1,
            ivar: ValID::IVar(0),
            body: &constant_access,
        };

        assert!(has_reuses(&tree));
    }

    #[test]
    fn perfect_nesting_requires_a_nonempty_access_body() {
        let context = Context::new();
        let access = access(AffineMap::new_constant(context.mlir_context(), 0), &[]);
        let accesses = [&access];
        let access_block = Tree::Block(&accesses);
        let empty: [&Tree<'_>; 0] = [];
        let empty_block = Tree::Block(&empty);

        assert!(is_perfectly_nested(&access_block));
        assert!(!is_perfectly_nested(&empty_block));
    }
}
