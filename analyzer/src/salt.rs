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

fn iteration_variable_to_poly(id: usize) -> Poly {
    let ring = IntegerRing::new();
    let atom = Atom::var(symbol!(format!("i{id}")));
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
#[allow(dead_code, reason = "keeps the distribution-only SALT API available")]
pub fn get_reuse_interval_distribution<'a, 'b: 'a>(
    tree: &Tree<'a>,
    reuse_factors: &mut HashMap<usize, Poly>,
    trip_counts: &mut HashMap<usize, Poly>,
    ref_count: usize,
    context: &AnalysisContext<'b>,
) -> HashMap<Poly, Poly> {
    let (distribution, _, _) = get_reuse_interval_distribution_with_instance_reports(
        tree,
        reuse_factors,
        trip_counts,
        ref_count,
        context,
    );
    distribution
}

#[derive(Clone, Debug)]
pub struct ReferenceInstanceReport {
    pub reference: String,
    pub distribution: Vec<(Poly, Poly)>,
    pub adjustment: Option<(Poly, Poly)>,
    pub adjustment_applied: bool,
}

/// Computes the reuse-interval distribution and the per-reference corrections needed by its
/// miss-ratio curve.
#[allow(dead_code, reason = "keeps the adjustment-only SALT API available")]
pub fn get_reuse_interval_distribution_with_adjustments<'a, 'b: 'a>(
    tree: &Tree<'a>,
    reuse_factors: &mut HashMap<usize, Poly>,
    trip_counts: &mut HashMap<usize, Poly>,
    ref_count: usize,
    context: &AnalysisContext<'b>,
) -> (HashMap<Poly, Poly>, Vec<(Poly, Poly)>) {
    let (distribution, adjustments, _) = get_reuse_interval_distribution_with_instance_reports(
        tree,
        reuse_factors,
        trip_counts,
        ref_count,
        context,
    );
    (distribution, adjustments)
}

/// Computes the kernel distribution, the adjustments actually passed to Denning recursion, and
/// a diagnostic report for every static memory-reference instance.
pub fn get_reuse_interval_distribution_with_instance_reports<'a, 'b: 'a>(
    tree: &Tree<'a>,
    reuse_factors: &mut HashMap<usize, Poly>,
    trip_counts: &mut HashMap<usize, Poly>,
    ref_count: usize,
    context: &AnalysisContext<'b>,
) -> (
    HashMap<Poly, Poly>,
    Vec<(Poly, Poly)>,
    Vec<ReferenceInstanceReport>,
) {
    let mut adjustments = Vec::new();
    let mut instance_reports = Vec::new();
    let mut distribution = get_reuse_interval_distribution_impl(
        tree,
        reuse_factors,
        trip_counts,
        ref_count,
        context,
        &mut adjustments,
        &mut instance_reports,
    );
    apply_restricted_constant_offset_group_reuse(
        tree,
        trip_counts,
        context,
        &mut distribution,
        &mut instance_reports,
    );
    adjustments = instance_reports
        .iter()
        .filter(|report| report.adjustment_applied)
        .filter_map(|report| report.adjustment.clone())
        .collect();
    (distribution, adjustments, instance_reports)
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum MemrefKey {
    Local(usize),
    Global(String),
}

fn memref_key(memref: &ValID) -> Option<MemrefKey> {
    match memref {
        ValID::Memref(id) => Some(MemrefKey::Local(*id)),
        ValID::Global(name) => Some(MemrefKey::Global(name.to_string())),
        _ => None,
    }
}

fn constant_poly_to_isize(poly: &Poly) -> Option<isize> {
    if !poly.numerator.is_constant() || !poly.denominator.is_constant() {
        return None;
    }
    let numerator = poly
        .numerator
        .get_constant()
        .to_string()
        .parse::<isize>()
        .ok()?;
    let denominator = poly
        .denominator
        .get_constant()
        .to_string()
        .parse::<isize>()
        .ok()?;
    if denominator == 0 || numerator % denominator != 0 {
        return None;
    }
    Some(numerator / denominator)
}

/// Extracts `A[i0 + c0, ..., in + cn]`'s constant offsets.
///
/// Requiring one unit-coefficient induction variable per map result deliberately excludes
/// tiled, strided, transposed, and symbolically shifted accesses from group reuse.
fn constant_offsets<'a>(
    map: AffineMap<'a>,
    operands: &'a [ValID],
    loop_ids: &[usize],
) -> Option<Vec<isize>> {
    let polys = convert_affine_map(map, operands).ok()?;
    if polys.len() != loop_ids.len() {
        return None;
    }
    polys
        .iter()
        .zip(loop_ids)
        .map(|(poly, loop_id)| {
            constant_poly_to_isize(&(poly.clone() - iteration_variable_to_poly(*loop_id)))
        })
        .collect()
}

fn add_distribution_mass(distribution: &mut HashMap<Poly, Poly>, interval: Poly, mass: Poly) {
    // Equivalent rational polynomials can retain different internal variable orderings and
    // consequently hash differently. Compare their normalized difference before inserting.
    if let Some(portion) = distribution.iter_mut().find_map(|(candidate, portion)| {
        (candidate.clone() - interval.clone())
            .numerator
            .is_zero()
            .then_some(portion)
    }) {
        *portion = &*portion + &mass;
    } else {
        distribution.insert(interval, mass);
    }
}

/// Applies restricted constant-offset group reuse for an untiled perfect loop nest.
///
/// For a fixed element, accesses `A[i + delta]` occur at iterations `i = x - delta`.
/// Sorting offsets in reverse lexicographic order therefore gives their chronological order.
/// Every access after the first replaces its ordinary cross-traversal reuse with the flattened
/// loop-body displacement from its predecessor. Boundary and alignment effects remain outside
/// this interior-pattern adjustment.
fn apply_restricted_constant_offset_group_reuse(
    tree: &Tree<'_>,
    trip_counts: &HashMap<usize, Poly>,
    context: &AnalysisContext<'_>,
    distribution: &mut HashMap<Poly, Poly>,
    instance_reports: &mut [ReferenceInstanceReport],
) {
    let mut loop_ids = Vec::new();
    let mut current = tree;
    loop {
        match current {
            Tree::For {
                ivar: ValID::IVar(id),
                step: 1,
                body,
                ..
            } => {
                loop_ids.push(*id);
                current = body;
            }
            Tree::Block([only]) => current = only,
            _ => break,
        }
    }

    // This implementation models the row-major, two-loop interior pattern described above.
    if loop_ids.len() != 2 {
        return;
    }
    let Tree::Block(trees) = current else {
        return;
    };
    if trees.is_empty() || !trees.iter().all(|tree| matches!(tree, Tree::Access { .. })) {
        return;
    }

    let reference_count = trees.len();
    let mut groups: Vec<(MemrefKey, Vec<(usize, Vec<isize>, bool)>)> = Vec::new();
    for (instance, tree) in trees.iter().enumerate() {
        let Tree::Access {
            memref,
            map,
            operands,
            is_write,
        } = tree
        else {
            continue;
        };
        let (Some(memref), Some(offsets)) = (
            memref_key(memref),
            constant_offsets(*map, operands, &loop_ids),
        ) else {
            continue;
        };
        if let Some((_, group)) = groups
            .iter_mut()
            .find(|(candidate, _)| *candidate == memref)
        {
            group.push((instance, offsets, *is_write));
        } else {
            groups.push((memref, vec![(instance, offsets, *is_write)]));
        }
    }

    let field = RationalPolynomialField::new(IntegerRing);
    let block_poly = Atom::var(symbol!("b")).to_rational_polynomial(
        &IntegerRing::new(),
        &IntegerRing::new(),
        None,
    );
    let reference_poly = constant_poly(reference_count as isize, context);
    let replacement_mass = field.div(&constant_poly(1, context), &(&reference_poly * &block_poly));
    let total_iterations = loop_ids
        .iter()
        .fold(constant_poly(1, context), |product, id| {
            &product
                * trip_counts
                    .get(id)
                    .expect("each stencil loop must have a trip count")
        });
    let long_interval =
        &reference_poly * &(&(&total_iterations + &constant_poly(1, context)) - &block_poly);

    for (_, mut instances) in groups {
        // A load immediately followed by a store to the same element is exact within-body reuse.
        // Its RI is the static memory-reference distance (one), rather than a loop displacement
        // multiplied by the number of references.
        instances.sort_by_key(|(instance, _, _)| *instance);
        if instances.len() == 2
            && instances[0].1 == instances[1].1
            && !instances[0].2
            && instances[1].2
            && instances[1].0 == instances[0].0 + 1
        {
            let load_instance = instances[0].0;
            let store_instance = instances[1].0;
            let store_distribution = instance_reports[store_instance].distribution.clone();
            for (interval, portion) in store_distribution {
                add_distribution_mass(distribution, interval, -portion);
            }

            let immediate_interval = constant_poly(1, context);
            let immediate_portion = field.div(&constant_poly(1, context), &reference_poly);
            add_distribution_mass(
                distribution,
                immediate_interval.clone(),
                immediate_portion.clone(),
            );
            distribution.retain(|_, portion| !portion.numerator.is_zero());

            instance_reports[load_instance].adjustment_applied = true;
            instance_reports[store_instance].distribution =
                vec![(immediate_interval.clone(), immediate_portion.clone())];
            instance_reports[store_instance].adjustment =
                Some((immediate_interval, immediate_portion));
            instance_reports[store_instance].adjustment_applied = false;
            continue;
        }

        // Two identical accesses are not a shifted-reference group. Repeated offsets also make
        // static operation order relevant, which this restricted adjustment intentionally avoids.
        if instances.iter().any(|(_, _, is_write)| *is_write) {
            continue;
        }
        instances.sort_by(|(_, left, _), (_, right, _)| right.cmp(left));
        if instances.len() < 2 || instances.windows(2).any(|pair| pair[0].1 == pair[1].1) {
            continue;
        }

        for (instance, _, _) in &instances {
            instance_reports[*instance].adjustment_applied = false;
        }
        let first_instance = instances[0].0;
        instance_reports[first_instance].adjustment =
            Some((long_interval.clone(), replacement_mass.clone()));
        instance_reports[first_instance].adjustment_applied = true;

        for pair in instances.windows(2) {
            let outer_delta = pair[0].1[0] - pair[1].1[0];
            let inner_delta = pair[0].1[1] - pair[1].1[1];
            let inner_trip_count = trip_counts
                .get(&loop_ids[1])
                .expect("the inner stencil loop must have a trip count");
            let displacement = &(&constant_poly(outer_delta, context) * inner_trip_count)
                + &constant_poly(inner_delta, context);
            let replacement_interval = &reference_poly * &displacement;

            add_distribution_mass(
                distribution,
                long_interval.clone(),
                -replacement_mass.clone(),
            );
            add_distribution_mass(
                distribution,
                replacement_interval.clone(),
                replacement_mass.clone(),
            );

            let instance = pair[1].0;
            add_instance_distribution_mass(
                &mut instance_reports[instance].distribution,
                long_interval.clone(),
                -replacement_mass.clone(),
            );
            add_instance_distribution_mass(
                &mut instance_reports[instance].distribution,
                replacement_interval.clone(),
                replacement_mass.clone(),
            );
            instance_reports[instance].adjustment =
                Some((replacement_interval, replacement_mass.clone()));
        }
    }
}

fn add_instance_distribution_mass(
    distribution: &mut Vec<(Poly, Poly)>,
    interval: Poly,
    mass: Poly,
) {
    if let Some((_, portion)) = distribution
        .iter_mut()
        .find(|(candidate, _)| (candidate.clone() - interval.clone()).numerator.is_zero())
    {
        *portion = &*portion + &mass;
    } else {
        distribution.push((interval, mass));
    }
    distribution.retain(|(_, portion)| !portion.numerator.is_zero());
}

fn get_reuse_interval_distribution_impl<'a, 'b: 'a>(
    tree: &Tree<'a>,
    reuse_factors: &mut HashMap<usize, Poly>,
    trip_counts: &mut HashMap<usize, Poly>,
    ref_count: usize,
    context: &AnalysisContext<'b>,
    curve_adjustments: &mut Vec<(Poly, Poly)>,
    instance_reports: &mut Vec<ReferenceInstanceReport>,
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

            get_reuse_interval_distribution_impl(
                body,
                reuse_factors,
                trip_counts,
                ref_count,
                context,
                curve_adjustments,
                instance_reports,
            )
        }
        Tree::Block(trees) => {
            let mut ri_dist: HashMap<Poly, Poly> = HashMap::new();
            for subtree in trees.iter() {
                let subtree_distribution = get_reuse_interval_distribution_impl(
                    subtree,
                    reuse_factors,
                    trip_counts,
                    trees.len(),
                    context,
                    curve_adjustments,
                    instance_reports,
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

            // Record the loop dimensions used by this access.
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
            // The first nonzero reference-vector boundary represents one imaginary final reuse.
            // Normalize its referenced trip-count product by all accesses, the number of
            // references, and the block size to obtain this reference's curve adjustment.
            let curve_adjustment = field.div(
                &field.div(
                    &field.div(&referenced_trip_product, &total_access),
                    &reference_count,
                ),
                &block_poly,
            );
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

            if let Some((highest_interval, _)) = reuse_intervals.first() {
                curve_adjustments.push((highest_interval.clone(), curve_adjustment.clone()));
            }

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
                        let portion = field.div(&portion, &reference_count);
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
            let adjustment = reuse_intervals
                .first()
                .map(|(interval, _)| (interval.clone(), curve_adjustment));
            instance_reports.push(ReferenceInstanceReport {
                reference: tree.to_string(),
                distribution: ri_dist
                    .iter()
                    .map(|(interval, portion)| (interval.clone(), portion.clone()))
                    .collect(),
                adjustment,
                adjustment_applied: true,
            });
            ri_dist
        }

        Tree::If { .. } => HashMap::new(),
    }
}

#[derive(Serialize)]
struct SaltResult {
    ri_values: Vec<String>,
    portions: Vec<String>,
    reference_instances: Vec<SerializedReferenceInstance>,
    total_count: String,
    miss_ratio_curve: MissRatioCurve,
    analysis_time: Duration,
}

#[derive(Serialize)]
struct SerializedReferenceInstance {
    instance: usize,
    reference: String,
    ri_values: Vec<String>,
    portions: Vec<String>,
    adjustment_ri: Option<String>,
    adjustment: Option<String>,
    adjustment_applied: bool,
}

fn poly_string(poly: &Poly) -> String {
    poly.to_expression()
        .printer(PrintOptions::file_no_namespace())
        .to_string()
}

pub fn substitute_instance_report_block_size(
    report: &ReferenceInstanceReport,
    block_size: usize,
) -> ReferenceInstanceReport {
    ReferenceInstanceReport {
        reference: report.reference.clone(),
        distribution: report
            .distribution
            .iter()
            .map(|(interval, portion)| {
                (
                    substitute_block_size(interval, block_size),
                    substitute_block_size(portion, block_size),
                )
            })
            .collect(),
        adjustment: report.adjustment.as_ref().map(|(interval, adjustment)| {
            (
                substitute_block_size(interval, block_size),
                substitute_block_size(adjustment, block_size),
            )
        }),
        adjustment_applied: report.adjustment_applied,
    }
}

pub fn create_instance_table(reports: &[ReferenceInstanceReport]) -> comfy_table::Table {
    use comfy_table::ContentArrangement;
    use comfy_table::modifiers::UTF8_ROUND_CORNERS;
    use comfy_table::presets::UTF8_FULL;

    let mut table = comfy_table::Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            "Instance",
            "Reference",
            "RI Value",
            "Portion",
            "Adjustment RI",
            "Adjustment",
            "Applied",
        ]);
    for (instance, report) in reports.iter().enumerate() {
        let (adjustment_ri, adjustment) = report
            .adjustment
            .as_ref()
            .map(|(interval, adjustment)| (poly_string(interval), poly_string(adjustment)))
            .unwrap_or_else(|| ("-".into(), "-".into()));
        for (interval, portion) in &report.distribution {
            table.add_row([
                instance.to_string(),
                report.reference.clone(),
                poly_string(interval),
                poly_string(portion),
                adjustment_ri.clone(),
                adjustment.clone(),
                if report.adjustment_applied {
                    "yes"
                } else {
                    "no"
                }
                .into(),
            ]);
        }
    }
    table
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
#[allow(dead_code, reason = "keeps the unadjusted conversion API available")]
pub fn get_ri_distribution(dist: &[(Poly, Poly)]) -> Vec<(isize, f64)> {
    get_adjusted_ri_distribution(dist, &[])
}

/// Converts a symbolic distribution into curve input and subtracts each reference's adjustment
/// from the portion at that reference's highest reuse interval. `MissRatioCurve::new` turns that
/// subtraction into an addition at that row and every following row.
pub fn get_adjusted_ri_distribution(
    dist: &[(Poly, Poly)],
    adjustments: &[(Poly, Poly)],
) -> Vec<(isize, f64)> {
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

    for (value, adjustment) in adjustments {
        let Ok(value) =
            value
                .to_expression()
                .evaluate(|x| x.to_f64(), &empty_const_map, &empty_symbol_map)
        else {
            continue;
        };
        let Ok(adjustment) = adjustment.to_expression().evaluate(
            |x| x.to_f64(),
            &empty_const_map,
            &empty_symbol_map,
        ) else {
            continue;
        };
        let value = value as isize;
        distro_map
            .entry(value)
            .and_modify(|portion| *portion -= adjustment)
            .or_insert(-adjustment);
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
    adjustments: &[(Poly, Poly)],
    instance_reports: &[ReferenceInstanceReport],
    accesses: usize,
    trip_counts: I,
    start_time: Instant,
) -> anyhow::Result<String>
where
    I: Iterator<Item = &'a Poly>,
{
    let total_count = get_total_count(accesses, trip_counts);
    let ri_values: Vec<String> = dist.iter().map(|(poly, _)| poly_string(poly)).collect();
    let portions: Vec<String> = dist.iter().map(|(_, poly)| poly_string(poly)).collect();
    let reference_instances = instance_reports
        .iter()
        .enumerate()
        .map(|(instance, report)| SerializedReferenceInstance {
            instance,
            reference: report.reference.clone(),
            ri_values: report
                .distribution
                .iter()
                .map(|(interval, _)| poly_string(interval))
                .collect(),
            portions: report
                .distribution
                .iter()
                .map(|(_, portion)| poly_string(portion))
                .collect(),
            adjustment_ri: report
                .adjustment
                .as_ref()
                .map(|(interval, _)| poly_string(interval)),
            adjustment: report
                .adjustment
                .as_ref()
                .map(|(_, adjustment)| poly_string(adjustment)),
            adjustment_applied: report.adjustment_applied,
        })
        .collect();
    let total_count = total_count
        .to_expression()
        .printer(PrintOptions::file_no_namespace())
        .to_string();
    let distribution = get_adjusted_ri_distribution(dist, adjustments).into_boxed_slice();
    let miss_ratio_curve = MissRatioCurve::new(&distribution);
    let analysis_time = start_time.elapsed();
    let result = SaltResult {
        ri_values,
        portions,
        reference_instances,
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

    fn ratio(numerator: isize, denominator: isize, context: &AnalysisContext<'_>) -> Poly {
        RationalPolynomialField::new(IntegerRing).div(
            &constant_poly(numerator, context),
            &constant_poly(denominator, context),
        )
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
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

    #[test]
    fn curve_adjustments_accumulate_at_tied_highest_intervals() {
        AnalysisContext::start(|context| {
            let dist = vec![
                (constant_poly(3, &context), ratio(1, 5, &context)),
                (constant_poly(4, &context), ratio(3, 10, &context)),
                (constant_poly(5, &context), ratio(1, 2, &context)),
            ];
            let adjustments = vec![
                (constant_poly(3, &context), ratio(1, 10, &context)),
                (constant_poly(4, &context), ratio(1, 20, &context)),
                (constant_poly(4, &context), ratio(1, 20, &context)),
            ];

            let distribution = get_adjusted_ri_distribution(&dist, &adjustments);
            let curve = MissRatioCurve::new(&distribution);
            let json = serde_json::to_value(curve).expect("the curve must serialize");
            let miss_ratio = json["miss_ratio"]
                .as_array()
                .expect("miss_ratio must be an array");

            // Relative to the unadjusted curve, row 3 gains 0.1. Rows 4 and 5 gain
            // that same 0.1 plus both tied 0.05 adjustments.
            assert_close(miss_ratio[1].as_f64().unwrap(), 0.9);
            assert_close(miss_ratio[2].as_f64().unwrap(), 0.7);
            assert_close(miss_ratio[3].as_f64().unwrap(), 0.2);
        });
    }

    #[test]
    fn reference_adjustment_is_normalized_by_unreferenced_trip_counts() {
        AnalysisContext::start(|context| {
            let mlir_context = context.mcontext();
            let inner_operands = [ValID::IVar(1)];
            let inner_access = access(
                AffineMap::new(mlir_context, 1, 0, &[AffineExpr::new_dim(mlir_context, 0)]),
                &inner_operands,
            );
            let inner_loop = Tree::For {
                lower_bound: AffineMap::new_constant(mlir_context, 0),
                upper_bound: AffineMap::new_constant(mlir_context, 5),
                lower_bound_operands: &[],
                upper_bound_operands: &[],
                step: 1,
                ivar: ValID::IVar(1),
                body: &inner_access,
            };
            let tree = Tree::For {
                lower_bound: AffineMap::new_constant(mlir_context, 0),
                upper_bound: AffineMap::new_constant(mlir_context, 7),
                lower_bound_operands: &[],
                upper_bound_operands: &[],
                step: 1,
                ivar: ValID::IVar(0),
                body: &inner_loop,
            };
            let mut reuse_factors = HashMap::new();
            let mut trip_counts = HashMap::new();

            let (_, adjustments) = get_reuse_interval_distribution_with_adjustments(
                &tree,
                &mut reuse_factors,
                &mut trip_counts,
                1,
                &context,
            );
            assert_eq!(adjustments.len(), 1);
            let adjustment = substitute_block_size(&adjustments[0].1, 1);
            let empty_const_map = AHashMap::<Atom, f64>::new();
            let empty_symbol_map = AHashMap::new();
            let adjustment = adjustment
                .to_expression()
                .evaluate(|value| value.to_f64(), &empty_const_map, &empty_symbol_map)
                .expect("the adjustment must be numeric");
            assert_close(adjustment, 1.0 / 7.0);
        });
    }

    #[test]
    fn five_point_stencil_uses_constant_offset_group_reuse() {
        AnalysisContext::start(|context| {
            let mlir_context = context.mcontext();
            let operands = [ValID::IVar(0), ValID::IVar(1)];
            let make_map = |outer_offset, inner_offset| {
                AffineMap::new(
                    mlir_context,
                    2,
                    0,
                    &[
                        AffineExpr::new_dim(mlir_context, 0)
                            + AffineExpr::new_constant(mlir_context, outer_offset),
                        AffineExpr::new_dim(mlir_context, 1)
                            + AffineExpr::new_constant(mlir_context, inner_offset),
                    ],
                )
            };

            let bottom = access(make_map(1, 0), &operands);
            let left = access(make_map(0, -1), &operands);
            let center = access(make_map(0, 0), &operands);
            let right = access(make_map(0, 1), &operands);
            let top = access(make_map(-1, 0), &operands);
            assert_eq!(
                constant_offsets(make_map(1, 0), &operands, &[0, 1]),
                Some(vec![1, 0])
            );
            let output = Tree::Access {
                memref: ValID::Memref(1),
                map: make_map(0, 0),
                operands: &operands,
                is_write: true,
            };
            let accesses = [&bottom, &left, &center, &right, &top, &output];
            let body = Tree::Block(&accesses);
            let inner = Tree::For {
                lower_bound: AffineMap::new_constant(mlir_context, 0),
                upper_bound: AffineMap::new_constant(mlir_context, 20),
                lower_bound_operands: &[],
                upper_bound_operands: &[],
                step: 1,
                ivar: ValID::IVar(1),
                body: &body,
            };
            let tree = Tree::For {
                lower_bound: AffineMap::new_constant(mlir_context, 0),
                upper_bound: AffineMap::new_constant(mlir_context, 10),
                lower_bound_operands: &[],
                upper_bound_operands: &[],
                step: 1,
                ivar: ValID::IVar(0),
                body: &inner,
            };
            let mut reuse_factors = HashMap::new();
            let mut trip_counts = HashMap::new();

            let (distribution, adjustments, instance_reports) =
                get_reuse_interval_distribution_with_instance_reports(
                    &tree,
                    &mut reuse_factors,
                    &mut trip_counts,
                    1,
                    &context,
                );
            let distribution = distribution
                .into_iter()
                .map(|(interval, portion)| {
                    (
                        substitute_block_size(&interval, 4),
                        substitute_block_size(&portion, 4),
                    )
                })
                .collect::<Vec<_>>();
            let numeric = get_ri_distribution(&distribution);

            assert_eq!(numeric.len(), 4, "distribution: {numeric:?}"); // Includes (0, 0).
            assert_eq!(numeric[1].0, 6);
            assert_close(numeric[1].1, 5.0 / 6.0);
            assert_eq!(numeric[2].0, 114); // 6 * TC_j - 6.
            assert_close(numeric[2].1, 1.0 / 12.0);
            assert_eq!(numeric[3].0, 1182); // 6 * TC_i * TC_j + 6 - 6b.
            assert_close(numeric[3].1, 1.0 / 12.0);

            let numeric_adjustments = adjustments
                .iter()
                .map(|(interval, adjustment)| {
                    let interval = substitute_block_size(interval, 4)
                        .to_expression()
                        .evaluate(
                            |value| value.to_f64(),
                            &AHashMap::<Atom, f64>::new(),
                            &AHashMap::new(),
                        )
                        .expect("the adjustment interval must be numeric")
                        as isize;
                    let adjustment = substitute_block_size(adjustment, 4)
                        .to_expression()
                        .evaluate(
                            |value| value.to_f64(),
                            &AHashMap::<Atom, f64>::new(),
                            &AHashMap::new(),
                        )
                        .expect("the adjustment must be numeric");
                    (interval, adjustment)
                })
                .fold(AHashMap::new(), |mut totals, (interval, adjustment)| {
                    *totals.entry(interval).or_insert(0.0) += adjustment;
                    totals
                });

            // Five A instances contribute one adjustment at their largest RI. B contributes
            // its own adjustment at the same RI: 2 / (6 * b) = 1 / 12 for b = 4.
            assert_eq!(numeric_adjustments.len(), 1);
            assert_close(numeric_adjustments[&1182], 1.0 / 12.0);

            assert_eq!(instance_reports.len(), 6);
            assert!(
                instance_reports
                    .iter()
                    .all(|report| report.adjustment.is_some())
            );
            assert_eq!(
                instance_reports[..5]
                    .iter()
                    .filter(|report| report.adjustment_applied)
                    .count(),
                1
            );
            assert!(instance_reports[5].adjustment_applied);

            let output_load = Tree::Access {
                memref: ValID::Memref(1),
                map: make_map(0, 0),
                operands: &operands,
                is_write: false,
            };
            let accesses = [&bottom, &left, &center, &right, &top, &output_load, &output];
            let body = Tree::Block(&accesses);
            let inner = Tree::For {
                lower_bound: AffineMap::new_constant(mlir_context, 0),
                upper_bound: AffineMap::new_constant(mlir_context, 20),
                lower_bound_operands: &[],
                upper_bound_operands: &[],
                step: 1,
                ivar: ValID::IVar(1),
                body: &body,
            };
            let tree = Tree::For {
                lower_bound: AffineMap::new_constant(mlir_context, 0),
                upper_bound: AffineMap::new_constant(mlir_context, 10),
                lower_bound_operands: &[],
                upper_bound_operands: &[],
                step: 1,
                ivar: ValID::IVar(0),
                body: &inner,
            };
            let mut reuse_factors = HashMap::new();
            let mut trip_counts = HashMap::new();
            let (distribution, adjustments, reports) =
                get_reuse_interval_distribution_with_instance_reports(
                    &tree,
                    &mut reuse_factors,
                    &mut trip_counts,
                    1,
                    &context,
                );
            let distribution = distribution
                .into_iter()
                .map(|(interval, portion)| {
                    (
                        substitute_block_size(&interval, 4),
                        substitute_block_size(&portion, 4),
                    )
                })
                .collect::<Vec<_>>();
            let numeric = get_ri_distribution(&distribution);

            assert_eq!(numeric.len(), 5, "distribution: {numeric:?}");
            assert_eq!(numeric[1].0, 1);
            assert_close(numeric[1].1, 1.0 / 7.0);
            assert_eq!(numeric[2].0, 7);
            assert_close(numeric[2].1, 5.0 / 7.0);
            assert_eq!(numeric[3].0, 133); // 7 * TC_j - 7.
            assert_close(numeric[3].1, 1.0 / 14.0);
            assert_eq!(numeric[4].0, 1379); // 7 * (TC_i * TC_j + 1 - b).
            assert_close(numeric[4].1, 1.0 / 14.0);

            assert_eq!(reports.len(), 7);
            assert_eq!(reports[6].distribution.len(), 1);
            let store_report = substitute_instance_report_block_size(&reports[6], 4);
            assert_eq!(
                constant_poly_to_isize(&store_report.distribution[0].0),
                Some(1)
            );
            assert_eq!(store_report.distribution[0].1, ratio(1, 7, &context));
            let store_adjustment = store_report
                .adjustment
                .as_ref()
                .expect("the unapplied store candidate must be reported");
            assert_eq!(constant_poly_to_isize(&store_adjustment.0), Some(1));
            assert_eq!(store_adjustment.1, ratio(1, 7, &context));
            assert!(!store_report.adjustment_applied);
            assert_eq!(adjustments.len(), 2); // One for A and one for the B load.
        });
    }
}
